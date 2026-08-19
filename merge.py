import json
import sys, os


indir = sys.argv[1]
# nametag = "mia_qwen25omni_generation_10_samples_minkpp_all_origmodel_capasr"
nametag = "mia_stepaudio_generation_10_samples_minkpp_all_origmodel_capasr"
outputfile = "{}.json".format(nametag)

alldata = []
# for filename in os.listdir(indir):
for i in range(24):
    filename = "{}_{}.json".format(nametag, i+1)
    # if outputfile not in filename and filename.startswith(nametag):
    with open(os.path.join(indir, filename)) as fin:
        data = json.load(fin)
    alldata.extend(data)

with open(os.path.join(indir, outputfile), "w") as fout:
    json.dump(alldata, fout, indent=4)