from typing import TYPE_CHECKING, Any, Dict, List, Optional

import sys
sys.path.append('./src')

from llmtuner.extras.logging import get_logger
from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.tuner.core import get_train_args, load_tokenizer

logger = get_logger(__name__)


def do_preprocess(args: Optional[Dict[str, Any]] = None):
    '''
    load data and preform preprocess and save
    '''
    logger.info(args)
    model_args, data_args, training_args, finetuning_args, generating_args, general_args = get_train_args(args)

    logger.info('model_args={}'.format(model_args))
    logger.info('data_args={}'.format(data_args))
    logger.info('training_args={}'.format(training_args))

    return
    dataset = get_dataset(model_args, data_args)
    tokenizer = load_tokenizer(model_args)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="pt")
    return


if __name__ == '__main__':
    do_preprocess()