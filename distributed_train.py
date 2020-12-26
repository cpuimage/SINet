from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from absl import app
import os
from data_loader import DatasetLoader
from compute_loss import data_term_loss
from model import Model
from adamp import AdamP


class Train(object):
    def __init__(self, batch_size, strategy, checkpoint_path, num_epochs, model, train_num_datasets, test_num_datasets,
                 train_len=None, num_gpu=1, save_epoch=1, checkpoint_dir="training", num_classes=2,
                 learning_rate=5e-4):
        self.num_epochs = num_epochs
        self.save_tensorboard_image = int(num_gpu) == 1
        self.checkpoint_path = checkpoint_path
        self.train_len = train_len
        self.batch_size = batch_size
        self.strategy = strategy
        self.num_gpu = int(num_gpu)
        self.train_epoch_step = (train_num_datasets // self.batch_size) - 1
        self.test_epoch_step = (test_num_datasets // self.batch_size) - 1
        self.save_epoch = int(save_epoch)
        self.model = model
        self.train_writer = tf.summary.create_file_writer('training')
        self.lr = self.multi_step_lr(initial_learning_rate=learning_rate, epochs=num_epochs)
        self.optimizer = AdamP(learning_rate=self.lr, weight_decay=1e-2)
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_dir, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self.epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
            global_step = self.epoch * self.train_epoch_step
            tf.get_logger().info("Latest checkpoint restored:{}".format(self.ckpt_manager.latest_checkpoint))
        else:
            global_step = 0
            self.epoch = 0
            tf.get_logger().info('Not restoring from saved checkpoint')
        self.global_step = tf.cast(global_step, tf.int64)
        self.train_acc_metric = tf.keras.metrics.MeanIoU(
            num_classes=num_classes + 1 if num_classes == 1 else num_classes, name='train_accuracy')
        self.test_acc_metric = tf.keras.metrics.MeanIoU(
            num_classes=num_classes + 1 if num_classes == 1 else num_classes, name='test_accuracy')
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

    def multi_step_lr(self, initial_learning_rate=5e-4, gamma=0.5, epochs=300):
        lr_steps_value = [initial_learning_rate]
        decay1 = epochs // 2
        decay2 = epochs - epochs // 6
        milestones = (decay1, decay2)
        for _ in range(len(milestones)):
            lr_steps_value.append(lr_steps_value[-1] * gamma)
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=milestones, values=lr_steps_value)

    def summary_loss(self):
        steps = self.global_step + self.optimizer.iterations
        with self.train_writer.as_default():
            tf.summary.scalar('train_loss', self.train_loss_metric.result(), step=steps)
            tf.summary.scalar('train_acc', self.train_acc_metric.result(), step=steps)
            tf.summary.scalar('test_loss', self.test_loss_metric.result(), step=steps)
            tf.summary.scalar('test_acc', self.test_acc_metric.result(), step=steps)

    def summary_image(self, mask, image, label, is_train=True, max_outputs=1):
        training_step = self.global_step + self.optimizer.iterations
        if is_train:
            tf.summary.image('train/image', image, max_outputs=max_outputs,
                             step=training_step)
            tf.summary.image('train/mask', mask, max_outputs=max_outputs,
                             step=training_step)
            tf.summary.image('train/label', label, max_outputs=max_outputs, step=training_step)
            tf.summary.image('train/output', mask * image, max_outputs=max_outputs,
                             step=training_step)
        else:
            tf.summary.image('test/image', image, max_outputs=max_outputs,
                             step=training_step)
            tf.summary.image('test/mask', mask, max_outputs=max_outputs,
                             step=training_step)
            tf.summary.image('test/label', label, max_outputs=max_outputs, step=training_step)
            tf.summary.image('test/output', mask * image, max_outputs=max_outputs,
                             step=training_step)

    @tf.function
    def compute_loss(self, label, matt_alpha):
        per_example_loss = data_term_loss(y_true=label, y_pred=matt_alpha)
        pred_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)
        return pred_loss

    def train_step(self, inputs, steps):
        image, label = inputs
        with tf.GradientTape() as tape:
            mask = self.model(image, training=True)
            pred_loss = self.compute_loss(label, mask)
        gradients = tape.gradient(pred_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        self.train_loss_metric(pred_loss)
        self.train_acc_metric(label, mask)
        if self.save_tensorboard_image:
            if tf.equal(tf.cast(steps + 1, tf.int64) % self.train_epoch_step, 0):
                with self.train_writer.as_default():
                    self.summary_image(mask, image, label, is_train=True)
        return steps + 1

    def test_step(self, inputs, steps):
        image, label = inputs
        mask = self.model(image, training=False)
        pred_loss = self.compute_loss(label, mask)
        self.test_acc_metric(label, mask)
        self.test_loss_metric(pred_loss)
        if self.save_tensorboard_image:
            if tf.equal(tf.cast(steps + 1, tf.int64) % self.test_epoch_step, 0):
                with self.train_writer.as_default():
                    self.summary_image(mask, image, label, is_train=False)
        return steps + 1

    def custom_loop(self, train_dist_dataset, test_dist_dataset, strategy):
        """Custom training and testing loop.

        Args:
          train_dist_dataset: Training dataset created using strategy.
          test_dist_dataset: Testing dataset created using strategy.
          strategy: Distribution strategy.
        """

        def distributed_train_epoch(train_iterator):
            per_replica_steps = tf.cast(0, tf.float64)
            for one_batch in train_iterator:
                per_replica_steps = strategy.run(
                    self.train_step, args=(one_batch, per_replica_steps))
                per_replica_steps = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_steps,
                                                    axis=None)
            return per_replica_steps

        def distributed_test_epoch(test_iterator):
            per_replica_steps = tf.cast(0, tf.float64)
            for one_batch in test_iterator:
                per_replica_steps = strategy.run(self.test_step, args=(one_batch, per_replica_steps))
                per_replica_steps = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_steps,
                                                    axis=None)
            return per_replica_steps

        distributed_train = tf.function(distributed_train_epoch)
        distributed_test = tf.function(distributed_test_epoch)
        for epoch in tf.range(self.epoch, self.num_epochs):
            self.train_epoch_step = tf.cast(distributed_train(train_dist_dataset) - 1, tf.int64)
            if epoch % self.save_epoch == 0:
                self.ckpt_manager.save()
                self.test_epoch_step = tf.cast(distributed_test(test_dist_dataset) - 1, tf.int64)
                self.summary_loss()
                description_str = 'Epoch:{}\n Train Loss:{}\t Train Accuracy:{}\t Test Loss:{}\t Test Accuracy:{}\t'.format(
                    epoch + 1,
                    self.train_loss_metric.result(),
                    self.train_acc_metric.result() * 100,
                    self.test_loss_metric.result(),
                    self.test_acc_metric.result() * 100,
                )
                tf.get_logger().info(description_str)
            if epoch != self.num_epochs - 1:
                self.train_loss_metric.reset_states()
                self.train_acc_metric.reset_states()
                self.test_loss_metric.reset_states()
                self.test_acc_metric.reset_states()
        self.ckpt_manager.save()


def main(num_epochs, buffer_size, batch_size, datasets_path=None, output_resolution=512,
         max_load_output_resolution=512, num_classes=2, num_gpu=None, use_tpu=False):
    physical_gpus = tf.config.experimental.list_physical_devices('GPU')
    if num_gpu is None:
        num_gpu = len(physical_gpus)
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    try:
        # TPU detection
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver() if use_tpu else None
    except ValueError:
        tpu = None
    # Select appropriate distribution strategy
    if use_tpu and tpu:
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        tf.get_logger().info('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    elif len(logical_gpus) > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=['/gpu:{}'.format(i) for i in range(num_gpu)]
        )
        tf.get_logger().info('Running on multiple GPUs.')
    elif len(logical_gpus) == 1:
        strategy = tf.distribute.get_strategy()
        tf.get_logger().info('Running on single GPU.')
    else:
        strategy = tf.distribute.get_strategy()
        tf.get_logger().info('Running on single CPU.')
    tf.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    tf.get_logger().info('num_classes: {}'.format(num_classes))
    tf.get_logger().info('batch_size: {}'.format(batch_size))
    tf.get_logger().info('output_resolution: {}'.format(output_resolution))
    checkpoint_path = "training/cp-{epoch:04d}-{step:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    dataset_loader = DatasetLoader(buffer_size=buffer_size, batch_size=batch_size * strategy.num_replicas_in_sync,
                                   output_resolution=output_resolution,
                                   max_load_output_resolution=max_load_output_resolution)
    train_dataset, test_dataset, train_num_datasets, test_num_datasets = dataset_loader.load(
        datasets_path=datasets_path, train_dir_name="train", test_dir_name="test")
    tf.get_logger().info("train_num_datasets:{}".format(train_num_datasets))
    tf.get_logger().info("test_num_datasets:{}".format(test_num_datasets))
    with strategy.scope():
        model = Model(output_resolution=output_resolution, num_classes=num_classes)
        train_len = tf.data.experimental.cardinality(train_dataset)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
        trainer = Train(batch_size=batch_size, strategy=strategy, num_epochs=num_epochs, model=model,
                        train_num_datasets=train_num_datasets,
                        test_num_datasets=test_num_datasets,
                        checkpoint_path=checkpoint_path,
                        train_len=train_len,
                        num_classes=num_classes,
                        num_gpu=num_gpu,
                        checkpoint_dir=checkpoint_dir)
        trainer.custom_loop(train_dist_dataset,
                            test_dist_dataset,
                            strategy)


def run_main(argv):
    """Passes the flags to main.

    Args:
      argv: argv
    """
    del argv
    kwargs = {
        'num_epochs': 5000,
        'buffer_size': 512,
        'batch_size': 32,
        'datasets_path': "./data",  # 'Directory to store the dataset'
        'output_resolution': 512,
        'max_load_output_resolution': 640,
        'num_classes': 1,
        'num_gpu': None,
    }
    main(**kwargs)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['TF_XLA_FLAGS'] = "--tf_xla_enable_xla_devices"
    app.run(run_main)
