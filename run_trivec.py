from tqdm import trange
import time
import os
import numpy as np
from datetime import datetime
from constants import PARAMS, DATA_CONST
from run_utils import create_parser
from src.negative_sampling import *
from src.knowledge_graph import KnowledgeGraph
from src.trivec_model import TriVec
from src.losses import NegativeSoftPlusLoss
from src.utils import switch_grad_mode, switch_model_mode
from run_utils import evaluation, evaluation_ranking


def make_dirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    parameters = PARAMS.copy()
    parameters['batch_size'] = args.batch_size
    parameters['embed_dim'] = args.embed_dim,
    parameters['epoch'] = args.epoch
    parameters['learning_rate'] = args.learning_rate
    parameters['regularization'] = args.regularization
    parameters['use_proteins'] = args.use_proteins
    parameters['reversed'] = args.reversed
    parameters['metrics_separately'] = args.metrics_separately
    parameters['random_val_neg_sampler'] = args.random_val_neg_sampler
    parameters['val_regenerate'] = args.val_regenerate

    if args.log:
        import neptune

        neptune.init(args.neptune_project)
        neptune_experiment_name = args.experiment_name
        neptune.create_experiment(name=neptune_experiment_name,
                                  params=parameters,
                                  upload_stdout=True,
                                  upload_stderr=True,
                                  send_hardware_metrics=True,
                                  upload_source_files='**/*.py')
        neptune.append_tag('pytorch')

        if args.gpu:
            neptune.append_tag('gpu')
        if args.use_proteins:
            neptune.append_tag('proteins')
        if args.reversed:
            neptune.append_tag('reversed')
        neptune.append_tag('real data')
        neptune.append_tag('trivec')
    use_cuda = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda" if args.gpu else "cpu")
    print(f'Use device: {device}')

    kg = KnowledgeGraph(data_path=DATA_CONST['work_dir'],
                        use_proteins=args.use_proteins,
                        use_proteins_on_validation=False,
                        use_reversed_edges=args.reversed)

    # Pos loaders
    train_pos_loader = data.DataLoader(
        torch.Tensor(np.array(kg.get_data_by_type('train'))),
        batch_size=parameters['batch_size'],
        shuffle=True)

    val_pos_loader = data.DataLoader(
        torch.Tensor(np.array(kg.get_data_by_type('val'))),
        batch_size=parameters['batch_size'],
        shuffle=False)

    test_pos_loader = data.DataLoader(
        torch.Tensor(np.array(kg.get_data_by_type('test'))),
        batch_size=parameters['batch_size'],
        shuffle=False)

    # Neg samplers
    train_neg_sampler = RandomNegativeSampler(
        kg.get_num_of_ent("train"), kg.get_num_of_ent('train'),
        neg_per_pos=parameters['num_of_neg_samples'])

    if args.random_val_neg_sampler:
        val_neg_sampler = RandomNegativeSampler(
            kg.get_num_of_ent("val"), kg.get_num_of_ent('val'),
            neg_per_pos=parameters['num_of_neg_samples'])
    else:
        val_neg_sampler = HonestNegativeSampler(
            kg.get_num_of_ent("val"), kg.get_num_of_ent('val'),
            kg.df_drug, neg_per_pos=1)

    test_neg_sampler = HonestNegativeSampler(
        kg.get_num_of_ent("test"), kg.get_num_of_ent('test'),
        kg.df_drug, neg_per_pos=1)

    # Neg datasets  (need to can regenerate neg examples)
    val_neg_data = NegDataset(
        torch.Tensor(np.array(kg.get_data_by_type('val'))),
        val_neg_sampler)
    test_neg_data = NegDataset(
        torch.Tensor(np.array(kg.get_data_by_type('test'))),
        test_neg_sampler)

    # Neg loaders
    val_neg_loader = data.DataLoader(val_neg_data,
                                     batch_size=parameters['batch_size'],
                                     shuffle=False)
    test_neg_loader = data.DataLoader(test_neg_data,
                                      batch_size=parameters['batch_size'],
                                      shuffle=False)

    # Model init
    model = TriVec(ent_total=kg.get_num_of_ent('train'),
                   rel_total=kg.get_num_of_rel('train'))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'],
                           amsgrad=True)
    loss_func = NegativeSoftPlusLoss()

    # Train
    model_save_path = (DATA_CONST['work_dir'] + DATA_CONST['save_path'] +
                       '/' + args.experiment_name)
    print('Train')
    print()
    for epoch in trange(args.epoch):
        train_epoch_losses = []
        for pos_triplets in train_pos_loader:
            switch_grad_mode(model, requires_grad=True)
            switch_model_mode(model, train=True)

            neg_triplets = train_neg_sampler(pos_triplets).to(device).long()
            pos_triplets = pos_triplets.to(device).long()
            pos_scores = model(pos_triplets)
            neg_scores = model(neg_triplets)
            loss = loss_func(pos_scores, neg_scores)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_epoch_losses.append(loss.item())

        if not epoch % args.print_progress_every:
            switch_grad_mode(model, requires_grad=False)
            switch_model_mode(model, train=False)

            val_loss, val_metrics = evaluation(
                val_pos_loader, val_neg_loader, model, device, loss_func,
                metrics_separately=args.metrics_separately)

            if args.log:
                neptune.log_metric("train_loss", np.mean(train_epoch_losses),
                                   timestamp=time.time())
                neptune.log_metric("val_loss", val_loss,
                                   timestamp=time.time())
                for metric, value in val_metrics.items():
                    neptune.log_metric(f'val_all_{metric}', value,
                                       timestamp=time.time())
            if args.val_regenerate:
                val_neg_data.regenerate()

        if not epoch % args.save_every:
            switch_grad_mode(model, requires_grad=False)
            switch_model_mode(model, train=False)
            t = datetime.now()
            path = model_save_path + '/' + t.strftime("%d_%m_%Y")
            make_dirs(path)
            path += '/' + str(epoch) + t.strftime("_%H_%M_%S") + '.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }, path)

    switch_grad_mode(model, requires_grad=False)
    switch_model_mode(model, train=False)
    t = datetime.now()
    path = model_save_path + '/' + t.strftime("%d_%m_%Y")
    make_dirs(path)
    path += '/' + str(args.epoch - 1) + t.strftime("_%H_%M_%S") + '.pt'
    torch.save({
        'epoch': args.epoch - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }, path)

    print('Test')
    #Loss and classic metrics
    test_loss, test_metrics = evaluation(
        test_pos_loader, test_neg_loader, model, device, loss_func,
        metrics_separately=args.metrics_separately)
    if args.log:
        neptune.log_metric("test_loss", test_loss, timestamp=time.time())
        for metric in test_metrics.keys():
            neptune.log_metric(f'test_all_{metric}', test_metrics[metric],
                               timestamp=time.time())
    if args.log:
        neptune.stop()
