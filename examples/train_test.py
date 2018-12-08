from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam

from transformer import get_or_create, save_config
from transformer.custom.callbacks import SGDRScheduler, LRFinder, WatchScheduler, LRSchedulerPerStep
from transformer.data_loader import DataLoader

if __name__ == '__main__':
    train_file_path = "../data/en2de.s2s.txt"
    valid_file_path = "../data/en2de.s2s.valid.txt"
    config_save_path = "../data/default-config.json"
    weights_save_path = "../models/weights.{epoch:02d}-{val_loss:.2f}.h5"
    init_weights_path = "../models/weights.36-3.04.h5"

    src_dict_path = "../data/dict_en.json"
    tgt_dict_path = "../data/dict_de.json"
    batch_size = 64
    epochs = 32

    # Data Loader
    data_loader = DataLoader(src_dictionary_path=src_dict_path,
                             tgt_dictionary_path=tgt_dict_path,
                             batch_size=batch_size)

    steps_per_epoch = 28998 // data_loader.batch_size
    validation_steps = 1014 // data_loader.batch_size

    config = {
        "src_vocab_size": data_loader.src_vocab_size,
        "tgt_vocab_size": data_loader.tgt_vocab_size,
        "model_dim": 512,
        "src_max_len": 70,
        "tgt_max_len": 70,
        "num_layers": 2,
        "num_heads": 8,
        "ffn_dim": 512,
        "dropout": 0.1
    }

    # Get transformer use config and load weights if exists.
    transformer = get_or_create(config,
                                optimizer=Adam(1e-3, 0.9, 0.98, epsilon=1e-9),
                                weights_path=init_weights_path)

    # save config
    save_config(transformer, config_save_path)

    ck = ModelCheckpoint(weights_save_path,
                         save_best_only=True,
                         save_weights_only=True,
                         monitor='val_loss',
                         verbose=0)
    log = TensorBoard(log_dir='../logs',
                      histogram_freq=0,
                      batch_size=data_loader.batch_size,
                      write_graph=True,
                      write_grads=False)

    # Use LRFinder to find effective learning rate
    lr_finder = LRFinder(1e-6, 1e-2, steps_per_epoch, epochs=3)  # => (3e-5, 5e-4)
    # lr_scheduler = WatchScheduler(lambda _, lr: lr / 2, min_lr=3e-5, max_lr=5e-4, watch="val_loss", watch_his_len=3)
    # lr_scheduler = LRSchedulerPerStep(512)
    lr_scheduler = SGDRScheduler(min_lr=4e-5, max_lr=5e-4, steps_per_epoch=steps_per_epoch,
                                 cycle_length=15,
                                 lr_decay=0.8,
                                 mult_factor=1.5)

    transformer.model.fit_generator(data_loader.generator(train_file_path),
                                    epochs=epochs,
                                    steps_per_epoch=steps_per_epoch,
                                    validation_data=data_loader.generator(valid_file_path),
                                    validation_steps=validation_steps,
                                    callbacks=[ck, log, lr_scheduler])

    # lr_finder.plot_loss()
    # lr_finder.plot_lr()
