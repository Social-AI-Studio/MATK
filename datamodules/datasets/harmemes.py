import os
import tqdm
import numpy as np
import pickle as pkl

from PIL import Image
from . import utils

from typing import List
from torch.utils.data import Dataset


INTENSITY_MAP = {
    'not harmful': 0, 
    'somewhat harmful': 1, 
    'very harmful': 2
}

TARGET_MAP = {
    'individual': 0, 
    'organization': 1, 
    'community': 2 , 
    'society': 3
}

class HarmemesBase(Dataset):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str]
    ):
        self.annotations = self._preprocess_annotations(annotation_filepath)
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        self.labels = labels

    def _preprocess_annotations(self, annotation_filepath: str):
        annotations = []

        # load the default annotations
        data = utils._load_jsonl(annotation_filepath)

        record_id = 0
        
        # translate labels into numeric values
        for record in tqdm.tqdm(data, desc="Preprocessing labels"):
            record["img"] = record.pop("image")
            record["intensity"] = INTENSITY_MAP[record["labels"][0]]
            record["target"] = TARGET_MAP[record["labels"][1]] if len(record["labels"]) > 1 else -1
            record["id"] = record_id
            record_id += 1
            annotations.append(record)
        
        return annotations

    def _load_auxiliary(self, auxiliary_dicts: dict):
        data = {}
        for key, filepath in tqdm.tqdm(auxiliary_dicts.items(), desc="Loading auxiliary info"):
            with open(filepath, "rb") as f:
                data[key] = pkl.load(f)

        return data

    def __len__(self):
        return len(self.annotations)


class FasterRCNNDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        feats_dict: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.feats_dict = feats_dict

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        text = record['text']
        image_id = record['img']
        id, _ = os.path.splitext(image_id)

        item = {
            'id': id,
            'image_id': image_id,
            'text': text,
            'roi_features': self.feats_dict[id]['roi_features'],
            'normalized_boxes': self.feats_dict[id]['normalized_boxes']
        }

        for l in self.labels:
            item[l] = record[l]

        return item


class ImagesDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        image_dir: str
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.image_dir = image_dir

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        image_filename = record['img']
        image_id, _ = os.path.splitext(image_filename)

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB") if image.mode != "RGB" else image

        item = {
            'id': record['id'],
            'image_id': image_id,
            'text': record['text'],
            'image': np.array(image),
            'image_path': image_path
        }

        for l in self.labels:
            item[l] = record[l]

        return item


class TextDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        input_template: str,
        output_template: str,
        label2word: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.input_template = input_template
        self.output_template = output_template
        self.label2word = label2word

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        # Format the input template
        input_kwargs = {"text": record['text']}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[f"{id:05}"]

        image_id, _ = os.path.splitext(record['img'])

        item = {
            'id': record["id"],
            'image_id': image_id,
            'text': self.input_template.format(**input_kwargs)
        }

        for l in self.labels:
            label = record[l]
            item[l] = self.output_template.format(label=self.label2word[label])

        return item

class MultimodalDataset():
    #mem, off, harm
    def __init__(self,opt,tokenizer,dataset,mode='train',few_shot_index=0):
        super(MultimodalDataset,self).__init__()
        self.opt=opt
        self.tokenizer = tokenizer
        self.mode=mode
        if self.opt['FEW_SHOT']:
            self.few_shot_index=str(few_shot_index)
            self.num_shots=self.opt['NUM_SHOTS']
            print ('Few shot learning setting for Iteration:',self.few_shot_index)
            print ('Number of shots:',self.num_shots)
        
        self.num_ans=self.opt['NUM_LABELS']
        #maximum length for a single sentence
        self.length=self.opt['LENGTH']
        #maximum length of the concatenation of sentences
        self.total_length=self.opt['TOTAL_LENGTH']
        self.num_sample=self.opt['NUM_SAMPLE']
        self.add_ent=self.opt['ADD_ENT']
        self.add_dem=self.opt['ADD_DEM']
        print ('Adding entity information?',self.add_ent)
        print ('Adding demographic information?',self.add_dem)
        self.fine_grind=self.opt['FINE_GRIND']
        print ('Using target information?',self.fine_grind)
        
        if opt['FINE_GRIND']:
            #target information
            if self.opt['DATASET']=='mem':
                self.label_mapping_word={0:'nobody',
                                         1:'race',
                                         2:'disability',
                                         3:'nationality',
                                         4:'sex',
                                         5:'religion'}
            elif self.opt['DATASET']=='harm':
                self.label_mapping_word={0:'nobody',
                                         1:'society',
                                         2:'individual',
                                         3:'community',
                                         4:'organization'}
                self.attack_list={'society':0,
                                  'individual':1,
                                  'community':2,
                                  'organization':3}
                self.attack_file=load_pkl(os.path.join(self.opt['DATA'],
                                                       'domain_splits','harm_trgt.pkl'))
            self.template="*<s>**sent_0*.*_It_was_targeting*label_**</s>*"
        else:
            self.label_mapping_word={0:self.opt['POS_WORD'],
                                     1:self.opt['NEG_WORD']}
            self.template="*<s>**sent_0*.*_It_was*label_**</s>*"
            
        self.label_mapping_id={}
        for label in self.label_mapping_word.keys():
            mapping_word=self.label_mapping_word[label]
            #add space already
            assert len(tokenizer.tokenize(' ' + self.label_mapping_word[label])) == 1
            self.label_mapping_id[label] = \
            tokenizer._convert_token_to_id(
                tokenizer.tokenize(' ' + self.label_mapping_word[label])[0])
            print ('Mapping for label %d, word %s, index %d' % 
                   (label,mapping_word,self.label_mapping_id[label]))
        #implementation for one template now
        
        
        self.template_list=self.template.split('*')
        print('Template:', self.template)
        print('Template list:',self.template_list)
        self.special_token_mapping = {
            '<s>': tokenizer.convert_tokens_to_ids('<s>'),
            '<mask>': tokenizer.mask_token_id, 
            '<pad>': tokenizer.pad_token_id, #1 for roberta
            '</s>': tokenizer.convert_tokens_to_ids('<\s>') 
        }
        
        if self.opt['DEM_SAMP']:
            print ('Using demonstration sampling strategy...')
            self.img_rate=self.opt['IMG_RATE']
            self.text_rate=self.opt['TEXT_RATE']
            self.samp_rate=self.opt['SIM_RATE']
            print ('Image rage for measuring CLIP similarity:',self.img_rate)
            print ('Text rage for measuring CLIP similarity:',self.text_rate)
            print ('Sampling from top:',self.samp_rate*100.0,'examples')
            self.clip_clean=self.opt['CLIP_CLEAN']
            clip_path=os.path.join(
                self.opt['CAPTION_PATH'],
                dataset,dataset+'_sim_scores.pkl')
            print ('Clip feature path:',clip_path)
            self.clip_feature=load_pkl(clip_path)
        
        self.support_examples=self.load_entries('train')
        print ('Length of supporting example:',len(self.support_examples))
        self.entries=self.load_entries(mode)
        if self.opt['DEBUG']:
            self.entries=self.entries[:128]
        self.prepare_exp()
        print ('The length of the dataset for:',mode,'is:',len(self.entries))

    def load_entries(self,mode):
        #only in training mode, in few-shot setting the loading will be different
        if self.opt['FEW_SHOT'] and mode=='train':
            path=os.path.join(self.opt['DATA'],
                              'domain_splits',
                              self.opt['DATASET']+'_'+str(self.num_shots)+'_'+self.few_shot_index+'.json')
        else:
            path=os.path.join(self.opt['DATA'],
                              'domain_splits',
                              self.opt['DATASET']+'_'+mode+'.json')
        data=read_json(path)
        cap_path=os.path.join(self.opt['CAPTION_PATH'],
                              self.opt['DATASET']+'_'+self.opt['PRETRAIN_DATA'],
                              self.opt['IMG_VERSION']+'_captions.pkl')
        captions=load_pkl(cap_path)
        entries=[]
        for k,row in enumerate(data):
            label=row['label']
            img=row['img']
            cap=captions[img.split('.')[0]][:-1]#remove the punctuation in the end
            sent=row['clean_sent']
            #remember the punctuations at the end of each sentence
            cap=cap+' . '+sent+' . '
            #whether using external knowledge
            if self.add_ent:
                cap=cap+' . '+row['entity']+' . '
            if self.add_dem:
                cap=cap+' . '+row['race']+' . '
            entry={
                'cap':cap.strip(),
                'label':label,
                'img':img
            }
            if self.fine_grind:
                if self.opt['DATASET']=='mem':
                    if label==0:
                        #[1,0,0,0,0,0]
                        entry['attack']=[1]+row['attack']
                    else:
                        entry['attack']=[0]+row['attack']
                elif self.opt['DATASET']=='harm':
                    if label==0:
                        #[1,0,0,0,0,0]
                        entry['attack']=[1,0,0,0,0]
                    else:
                        attack=[0,0,0,0,0]
                        attack_idx=self.attack_list[self.attack_file[img]]+1
                        attack[attack_idx]=1
                        entry['attack']=attack
            entries.append(entry)
        return entries
    
    def enc(self,text):
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def prepare_exp(self):
        ###add sampling
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in tqdm(range(self.num_sample)):
            for query_idx in range(len(self.entries)):
                if self.opt['DEM_SAMP']:
                    #filter dissimilar demonstrations
                    candidates= [support_idx for support_idx in support_indices
                                 if support_idx != query_idx or self.mode != "train"]
                    sim_score=[]
                    count_each_label = {label: 0 for label in range(self.opt['NUM_LABELS'])}
                    context_indices=[]
                    clip_info_que=self.clip_feature[self.entries[query_idx]['img']]
                    
                    #similarity computation
                    for support_idx in candidates:
                        img=self.support_examples[support_idx]['img']
                        #this cost a lot of computation
                        #unnormalized: the same scale -- 512 dimension
                        if self.clip_clean:
                            img_sim=clip_info_que['clean_img'][img]
                        else:
                            img_sim=clip_info_que['img'][img]
                        text_sim=clip_info_que['text'][img]
                        total_sim=self.img_rate*img_sim+self.text_rate*text_sim
                        sim_score.append((support_idx,total_sim))
                    sim_score.sort(key=lambda x: x[1],reverse=True)
                    
                    #top opt['SIM_RATE'] entities for each label
                    num_valid=int(len(sim_score)//self.opt['NUM_LABELS']*self.samp_rate)
                    """
                    if self.opt['DEBUG']:
                        print ('Valid for each class:',num_valid)
                    """
                    
                    for support_idx, score in sim_score:
                        cur_label=self.support_examples[support_idx]['label']
                        if count_each_label[cur_label]<num_valid:
                            count_each_label[cur_label]+=1
                            context_indices.append(support_idx)
                else: 
                    #exclude the current example during training
                    context_indices = [support_idx for support_idx in support_indices
                                       if support_idx != query_idx or self.mode != "train"]
                #available indexes for supporting examples
                self.example_idx.append((query_idx, context_indices, sample_idx))

    def select_context(self, context_examples):
        """
        Select demonstrations from provided examples.
        """
        num_labels=self.opt['NUM_LABELS']
        max_demo_per_label = 1
        counts = {k: 0 for k in range(num_labels)}
        if num_labels == 1:
            # Regression
            counts = {'0': 0, '1': 0}
        selection = []
        """
        # Sampling strategy from LM-BFF
        if self.opt['DEBUG']:
            print ('Number of context examples available:',len(context_examples))
        """
        order = np.random.permutation(len(context_examples))
        for i in order:
            label = context_examples[i]['label']
            if num_labels == 1:
                # Regression
                #No implementation currently
                label = '0' if\
                float(label) <= median_mapping[self.args.task_name] else '1'
            if counts[label] < max_demo_per_label:
                selection.append(context_examples[i])
                counts[label] += 1
            if sum(counts.values()) == len(counts) * max_demo_per_label:
                break
        
        assert len(selection) > 0
        return selection
    
    def process_prompt(self, examples, 
                       first_sent_limit, other_sent_limit):
        if self.fine_grind:
            prompt_arch=' It was targeting '
        else:
            prompt_arch=' It was '
        #currently, first and other limit are the same
        input_ids = []
        attention_mask = []
        mask_pos = None # Position of the mask token
        concat_sent=""
        for segment_id, ent in enumerate(examples):
            #tokens for each example
            new_tokens=[]
            if segment_id==0:
                #implementation for the querying example
                new_tokens.append(self.special_token_mapping['<s>'])
                length=first_sent_limit
                temp=prompt_arch+'<mask>'+' . </s>'
            else:
                length=other_sent_limit
                if self.fine_grind:
                    if ent['label']==0:
                        label_word=self.label_mapping_word[0]
                    else:
                        attack_types=[i for i, x in enumerate(ent['attack']) if x==1]
                        #only for meme
                        if len(attack_types)==0:
                            attack_idx=random.randint(1,5)
                        #randomly pick one
                        #already padding nobody to the head of the list
                        else:
                            order=np.random.permutation(len(attack_types))
                            attack_idx=attack_types[order[0]]
                        label_word=self.label_mapping_word[attack_idx]
                else:
                    label_word=self.label_mapping_word[ent['label']]
                temp=prompt_arch+label_word+' . </s>'
            new_tokens+=self.enc(' '+ent['cap'])
            #truncate the sentence if too long
            new_tokens=new_tokens[:length]
            new_tokens+=self.enc(temp)
            whole_sent=' '+ent['cap']+temp
            concat_sent+=whole_sent
        
            #update the prompts
            input_ids+=new_tokens
            attention_mask += [1 for i in range(len(new_tokens))]
        """
        if self.opt['DEBUG'] and self.opt['DEM_SAMP']==False:
            print (concat_sent)
        """
        while len(input_ids) < self.total_length:
            input_ids.append(self.special_token_mapping['<pad>'])
            attention_mask.append(0)
        if len(input_ids) > self.total_length:
            input_ids = input_ids[:self.total_length]
            attention_mask = attention_mask[:self.total_length]
        mask_pos = [input_ids.index(self.special_token_mapping['<mask>'])]
        
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < self.total_length
        result = {'input_ids': input_ids,
                  'sent':'<s>'+concat_sent,
                  'attention_mask': attention_mask,
                  'mask_pos': mask_pos}
        return result
       
    def __getitem__(self,index):
        #query item
        entry=self.entries[index]
        #bootstrap_idx --> sample_idx
        query_idx, context_indices, bootstrap_idx = self.example_idx[index]
        #one example from each class
        supports = self.select_context(
            [self.support_examples[i] for i in context_indices])
        exps=[]
        exps.append(entry)
        exps.extend(supports)
        prompt_features = self.process_prompt(
            exps,
            self.length,
            self.length
        )
            
        vid=entry['img']
        #label=torch.tensor(self.label_mapping_id[entry['label']])
        label=torch.tensor(entry['label'])
        target=torch.from_numpy(np.zeros((self.num_ans),dtype=np.float32))
        target[label]=1.0
        
        cap_tokens=torch.Tensor(prompt_features['input_ids'])
        mask_pos=torch.LongTensor(prompt_features['mask_pos'])
        mask=torch.Tensor(prompt_features['attention_mask'])
        batch={
            'sent':prompt_features['sent'],
            'mask':mask,
            'img':vid,
            'target':target,
            'cap_tokens':cap_tokens,
            'mask_pos':mask_pos,
            'label':label
        }
        if self.fine_grind:
            batch['attack']=torch.Tensor(entry['attack'])
        #print (batch)
        return batch
        
    def __len__(self):
        return len(self.entries)