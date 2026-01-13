import pickle
import os
import config
from record_split import RecordSplit
from parameter_parser import parameter_parser

def main():
    parser = parameter_parser()
    args = parser.parse_args()

    # دیتاست از parser می‌آید
    dataset = args.dataset_name.lower()

    # تعداد نمونه‌ها
    if dataset == "mnist":
        num_records = 60000
    elif dataset == "cifar10":
        num_records = 50000
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    splitter = RecordSplit(num_records, args)
    splitter.split_shadow_target()
    splitter.sample_records(args.unlearning_method)

    os.makedirs(config.SPLIT_INDICES_PATH, exist_ok=True)
    save_path = config.SPLIT_INDICES_PATH + dataset

    with open(save_path, 'wb') as f:
        pickle.dump(splitter, f)

    print("✅ Split saved to:", save_path)

if __name__ == "__main__":
    main()
