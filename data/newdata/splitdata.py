import json


njobs = 24

# dataname = "caption_asr_testset"
# dataname = "testdata_capasr"
dataname = "testdata_QA_train"
with open("{}.json".format(dataname)) as fin:
    data = json.load(fin)

stepsize = len(data) // njobs + 1
for i in range(njobs):
    with open("split/{}_{}jobs_split{}.json".format(dataname, njobs, i+1), "w") as fout:
        json.dump(data[i*stepsize:(i+1)*stepsize], fout, indent=4)