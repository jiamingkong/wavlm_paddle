import yaml

config = yaml.load(open("wavlm/test_conf.yaml", "r"), Loader=yaml.FullLoader)

from wavlm import 