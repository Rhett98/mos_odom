import yaml
import torch
from tqdm import tqdm
from utility.dataset.kitti.parser_multiscan import Parser

ARCH = yaml.safe_load(open('config/arch/mos-test.yml', 'r'))
DATA = yaml.safe_load(open('config/data/local-test.yaml', 'r'))
data = '../dataset'
epsilon_w = ARCH["train"]["epsilon_w"]
parser = Parser(root=data,
                        train_sequences=DATA["split"]["train"],
                        valid_sequences=DATA["split"]["valid"],
                        test_sequences=None,
                        split='train',
                        labels=DATA["labels"],
                        color_map=DATA["color_map"],
                        learning_map=DATA["learning_map"],
                        learning_map_inv=DATA["learning_map_inv"],
                        sensor=ARCH["dataset"]["sensor"],
                        max_points=ARCH["dataset"]["max_points"],
                        batch_size=ARCH["train"]["batch_size"],
                        workers=ARCH["train"]["workers"],
                        gt=True,
                        shuffle_train=True)
loader = parser.get_train_set()
assert len(loader) > 0
# for i, (proj_in, proj_mask,proj_labels, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints,trans,rot) in enumerate(loader):
#     print(trans[0].data)
#     print(rot[0].data)
#     print("*******")
content = torch.zeros(parser.get_n_classes(), dtype=torch.float)
for cl, freq in DATA["content"].items():
    x_cl = parser.to_xentropy(cl)  # map actual class to xentropy class
    content[x_cl] += freq
loss_w = 1 / (content + epsilon_w)  # get weights
for x_cl, w in enumerate(loss_w):  # ignore the ones necessary to ignore
    if DATA["learning_ignore"][x_cl]:
        # don't weigh
        loss_w[x_cl] = 0
print("Loss weights from content: ", loss_w.data)

# set train and valid evaluator
ignore_class = []
for i, w in enumerate(loss_w):
    if w < 1e-10:
        ignore_class.append(i)
        print("Ignoring class ", i, " in IoU evaluation")