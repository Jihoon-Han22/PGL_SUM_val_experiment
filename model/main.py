# -*- coding: utf-8 -*-
from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    """ Main function that sets the data loaders; trains and evaluates the model."""
    config = get_config(mode='train')
    val_config = get_config(mode='val')
    test_config = get_config(mode='test')

    print(config)
    print(val_config)
    print(test_config)

    f = open(str(config.root_dir) + '/configuration.txt', 'w')
    f.write(str(config) + '\n\n\n\n')
    f.write(str(val_config) + '\n\n\n\n')
    f.write(str(test_config) + '\n\n\n\n')
    f.flush()
    f.close()

    print('Currently selected split_index:', config.split_index)
    train_loader = get_loader(mode='train', val_size = 2000, batch_size=config.batch_size, video_type='yt8m')
    val_loader = get_loader(mode='val', val_size = 2000, batch_size=1, video_type='yt8m')
    test_loader = get_loader(mode='test', val_size = 2000, batch_size=1, video_type='yt8m')
    #test_summe_loader = get_loader(mode='test', batch_size=1, video_type='summe')
    #test_tvsum_loader = get_loader(mode='test', batch_size=1, video_type='tvsum')
    
    solver = Solver(config, train_loader, val_loader, test_loader)

    solver.build()
    test_model_ckpt_path = None

    if config.train:
        # TODO: Discuss whether it is necessary
        # solver.evaluate(-1)	 # evaluates the summaries using the initial random weights of the network
        best_model_path = solver.train()
        test_model_ckpt_path = best_model_path
    
    if config.test:
        solver.test(test_model_ckpt_path)

# tensorboard --logdir '../PGL-SUM/Summaries/PGL-SUM/'
