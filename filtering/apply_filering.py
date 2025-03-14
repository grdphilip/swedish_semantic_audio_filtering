from filtering import FilteringFramework
from model_utils import merge_conf
import argparse

def main(args):
    
    base_conf_path = 'configs/base_config.yaml'
    dataset_conf_path = 'configs/dataset.yaml'
    model_conf_path = 'configs/model.yaml'
    config = merge_conf(base_conf_path, dataset_conf_path, model_conf_path)

    filtering = FilteringFramework(config, pretrained_model_path='save/experiments/model1/best_model.pth.tar')

    filtering.run(data_manifest_path=args.data_manifest_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_manifest_path", type=str, default='data/common_voice_16_0_train_manifest.json')
    args = parser.parse_args()

    main(args)
    
#python apply_filering.py data/syndata_fb_train_manifest.json