from dataset_builder_loader.data_loader import DataLoader
from sleep_stage_config import Config
import argparse
import sys


def main(args):
    config = Config()
    data_loader = DataLoader(config, args.modality, args.num_classes, args.seq_len)
    data_loader.build_windowed_cache_data(args.seq_len)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--modality', type=str, default="all", help=r"The modality to use. building up cache only "
                                                                    r"works when modality is 'all', as other modalities"
                                                                    r"included in the cache data file",
                        choices={"all"})
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes or labels',
                        choices={2, 3, 4, 5})
    parser.add_argument('--seq_len', type=int, default=100, help='number of classes or labels',
                        choices={20, 50, 100})
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

