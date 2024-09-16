import torch
import cherrypy
import cherrypy_cors
from argparse import ArgumentParser
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


class MyWebService(object):
    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def index(self):
        try:
            data = cherrypy.request.json
        except TypeError:
            return {'error': 'invalid input'}
        representation = calculate_embedding(data['input'])
        return {'embedding': representation.tolist()}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def arg_loader():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='text2vec_large')
    parser.add_argument('--port', type=int, default=8500)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    return parser.parse_args()


def calculate_embedding(sentences):
    if args.model in ['text2vec_large']:
        result = []
        model.eval()
        with torch.no_grad():
            i = 0
            while i < len(sentences):
                tokens = tokenizer(sentences[i:i + args.batch_size], padding=True, truncation=True, max_length=512,
                                   return_tensors='pt')
                if torch.cuda.is_available():
                    tokens = {k: v.cuda(args.gpu) for k, v in tokens.items()}
                outputs = model(**tokens)

                if args.model == 'text2vec_large':
                    outputs = mean_pooling(outputs, tokens['attention_mask'])
                # elif args.model == 'gte_base':
                #    outputs = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError

                outputs = torch.nn.functional.normalize(outputs)
                result.append(outputs)
                i += args.batch_size
        result = torch.cat(result, dim=0)
    else:
        result = model.encode(sentences, batch_size=args.batch_size, normalize_embeddings=True)
    return result


if __name__ == '__main__':
    args = arg_loader()
    print('================')
    for key, value in vars(args).items():
        print(key, value)
    print('================')

    if args.model == 'text2vec_large':
        tokenizer = BertTokenizer.from_pretrained('GanymedeNil/text2vec-large-chinese')
        model = BertModel.from_pretrained('GanymedeNil/text2vec-large-chinese')
    elif args.model == 'gte_base':
        model = SentenceTransformer('thenlper/gte-base-zh')
    elif args.model == 'gte_large':
        model = SentenceTransformer('thenlper/gte-large-zh')
    elif args.model == 'zpoint_large':
        model = SentenceTransformer('iampanda/zpoint_large_embedding_zh')
    elif args.model == 'yinka':
        model = SentenceTransformer('Classical/Yinka')
    elif args.model == 'acge':
        model = SentenceTransformer('aspire/acge_text_embedding')
    elif args.model == 'stella_mrl_large':
        model = SentenceTransformer('infgrad/stella-mrl-large-zh-v3.5-1792d')
    elif args.model == 'stella_large':
        model = SentenceTransformer('infgrad/stella-large-zh-v3-1792d')
    elif args.model == 'gte_large_en':
        model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        print('moving model to gpu %d' % args.gpu)
        model = model.cuda(args.gpu)

    print('starting service...')
    cherrypy_cors.install()
    config = {
        'global': {
            'server.socket_host': '0.0.0.0',
            'server.socket_port': args.port,
            'cors.expose.on': True
        }
    }
    cherrypy.config.update(config)
    cherrypy.quickstart(MyWebService(), '/', config)

# python3 service.py --model zpoint_large --gpu 0 --port 8500 --batch_size 128
