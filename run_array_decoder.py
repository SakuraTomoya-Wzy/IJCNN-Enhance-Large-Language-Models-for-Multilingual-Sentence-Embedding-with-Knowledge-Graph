import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)

# os.environ["HF_DATASETS_OFFLINE"] = "1"  # 1 for offline
# os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 1 for offline
# os.environ["TRANSFORMERS_CACHE"] = "/gpfswork/rech/six/commun/models"
# os.environ["HF_DATASETS_CACHE"] = "/gpfswork/rech/six/commun/datasets"
# os.environ["HF_MODULES_CACHE"] = "/gpfswork/rech/six/commun/modules"
# os.environ["HF_METRICS_CACHE"] = "/gpfswork/rech/six/commun/metrics"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from copy import deepcopy


TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    # "ArxivClusteringP2P",
    # "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    # "RedditClustering",
    # "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions", #KeyError: 'Franklin U722 USB modem signal strength'
    "TwitterSemEval2015", #KeyError: 'All the home alones watching 8 mile'
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    # "AskUbuntuDupQuestions",
    # "MindSmallReranking",
    # "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    # "ClimateFEVER",
    # "CQADupstackAndroidRetrieval",
    # "CQADupstackEnglishRetrieval",
    # "CQADupstackGamingRetrieval",
    # "CQADupstackGisRetrieval",
    # "CQADupstackMathematicaRetrieval",
    # "CQADupstackPhysicsRetrieval",
    # "CQADupstackProgrammersRetrieval",
    # "CQADupstackStatsRetrieval",
    # "CQADupstackTexRetrieval",
    # "CQADupstackUnixRetrieval",
    # "CQADupstackWebmastersRetrieval",
    # "CQADupstackWordpressRetrieval",
    # "DBPedia",
    # "FEVER",
    # "FiQA2018",
    # "HotpotQA",
    # "MSMARCO",
    "NFCorpus",
    # "NQ",
    # "QuoraRetrieval",
    # "SCIDOCS",
    "SciFact",
    # "Touche2020",
    # "TRECCOVID",
]

TASK_LIST_STS = [
    # "BIOSSES", #s2s 但是很长
    # "SICK-R",
    # "STS12",
    # "STS13",
    # "STS14",
    # "STS15",
    # "STS16",
    "STS17",
    # "STS22", #p2p
    # "STSBenchmark",
    # "SummEval",
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

from peft import PeftModel


class DecoderWrapper:
    def __init__(
        self,
        modelpath="princeton-nlp/sup-simcse-bert-base-uncased",
        lora_weight="",
        # mask_embedding_sentence_template='This_sentence_:_"of_or_relating_to_tutors_or_tutoring."_means_in_one_word:Tutorial.This_sentence_:_"*sent_0*"_means_in_one_word:',
        mask_embedding_sentence_template='This_sentence_:_"*sent_0*"_means_in_one_word:',
        avg=False,
        bf16 = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            modelpath, trust_remote_code=True
        )
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        # self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.tokenizer.padding_side = "left"  # Allow batched inference
        self.model = AutoModelForCausalLM.from_pretrained(
            modelpath,
            output_hidden_states=True,
            trust_remote_code=True,
            torch_dtype=torch.float16 if bf16 == False else torch.bfloat16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(
           self.model,
           lora_weight,
           torch_dtype=torch.float16 if bf16 == False else torch.bfloat16,
           device_map={"": 0},
        )
        self.model.eval()
        self.mask_embedding_sentence_template = mask_embedding_sentence_template
        self.avg = avg
        print(self.mask_embedding_sentence_template)

    def encode(self, raw_sentences, batch_size=32, **kwargs):
        """Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        # if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
        #     batch = [[word.decode("utf-8") for word in s] for s in batch]

        # sentences = [" ".join(s) for s in batch]
        # input_sentences = [" ".join(s) for s in batch]
        # if max_length == 500:
        #     sentences = [
        #         self.tokenizer.decode(
        #             self.tokenizer.encode(s, add_special_tokens=False)[:max_length]
        #         )
        #         for s in sentences
        #     ]
        #     max_length = 512
        max_length = 1024
        sentences = deepcopy(raw_sentences)

        if (
            # args.mask_embedding_sentence
            # and
            self.mask_embedding_sentence_template
            is not None
        ):
            # *cls*_This_sentence_of_"*sent_0*"_means*mask*.*sep+*
            template = self.mask_embedding_sentence_template
            template = (
                template.replace("_", " ").replace("*sep+*", "").replace("*cls*", "")
            )

            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in ".?\"'":
                    s += "."
                s = s.replace('"', "'")
                if len(s) > 0 and "?" == s[-1]:
                    s = s[:-1] + "."
                sentences[i] = template.replace("*sent 0*", s).strip()

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            batch = self.tokenizer.batch_encode_plus(
                sentences_batch,
                return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=max_length is not None,
            )
            # Move to the correct device
            # for k in batch:
            #     batch[k] = batch[k].to(self.device) if batch[k] is not None else None
            #    # Get raw embeddings
            batch = {k: v.to(self.device) for k,v in batch.items()}
            with torch.no_grad():
                hidden_states = self.model(
                    output_hidden_states=True, return_dict=True, **batch
                ).hidden_states
                if self.avg:
                    last_layer = hidden_states[-1]
                    attention_mask = (
                        batch["attention_mask"].unsqueeze(-1).expand(last_layer.shape)
                    )
                    outputs = (last_layer * attention_mask).mean(1)
                else:
                    outputs = hidden_states[-1][:, -1, :]

                if outputs.dtype == torch.bfloat16:
                    # bfloat16 not support for .numpy()
                    outputs = outputs.float()
            all_embeddings.extend(outputs.float().cpu().numpy())
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        return all_embeddings

        # 我不确定他这个sentences_sorted为什么要sorted
        # all_embeddings = []
        # length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        # sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        # for start_index in range(0, len(sentences), batch_size):
        #     sentences_batch = sentences_sorted[start_index:start_index+batch_size]
        #     inputs = self.tokenizer(sentences_batch, padding=True, truncation=True, return_tensors="pt")
        #     inputs = {k: v.to(self.device) for k,v in inputs.items()}
        #     # Get the embeddings
        #     with torch.no_grad():
        #         embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        #     all_embeddings.extend(embeddings.cpu().numpy())
        # all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        # return all_embeddings


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=32)
    args = parser.parse_args()
    return args


def main(args):
    # model = DecoderWrapper(modelpath='/root/hdd/llm/Mistral-7B-v0.1', lora_weight='/root/hdd/scaling_sentemb/mis-lora/checkpoint-600',bf16=True)
    model = DecoderWrapper(modelpath='/root/hdd/llm/Yi-6B', lora_weight="/root/hdd/scaling_sentemb/yi-lora/checkpoint-950",bf16=True)
    # model = DecoderWrapper(modelpath='/root/hdd/llm/falcon-rw-1b', lora_weight=args.lora,bf16=True)

    # if args.taskname is not None:
    #     task = args.taskname
    #     eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
    #     model_name = args.modelpath.split("/")[-1].split("_")[-1]
    #     evaluation = MTEB(tasks=[task], task_langs=[args.lang], eval_splits=eval_splits)
    #     evaluation.run(
    #         model, output_folder=f"results/{model_name}", batch_size=args.batchsize
    #     )
    #     exit()

    # for task in TASK_LIST[args.startid : args.endid]:
    for task in TASK_LIST_STS:
        print("Running task: ", task)
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        # model_name = 'falcon_'+args.lora.split('-')[-1]
        model_name = 'yi_multi'#+args.lora.split('-')[-1]
        # evaluation = MTEB(tasks=[task], task_langs=[args.lang])
        evaluation = MTEB(tasks=[task], task_langs=["es-en","fr-en","en-de","it-en","nl-en"])
        evaluation.run(
            model,
            output_folder=f"results/{model_name}",
            batch_size=args.batchsize,
            eval_splits=eval_splits,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
