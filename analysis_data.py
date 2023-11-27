fname ="/home/jingy/ABSA/Data_analysis/bert_spc_restaurant_analysis/bert_spc_asp1_labeled_raw.txt"

fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
lines = fin.readlines()
fin.close()
over_aspect = 0
over_predict = 0
POS, NEU, NEG = 0, 0, 0
POS1, NEU1, NEG1 = 0, 0, 0
total_samples = 0
for i in range(0, len(lines), 6):
    total_samples += 1
    text_raw = lines[i]
    over_polarity = lines[i + 1]
    aspect_raw = lines[i + 2]
    aspect_polarity = lines[i + 3]
    predict = lines[i + 4]
    if int(over_polarity) == int(aspect_polarity):
        over_aspect += 1
    if int(over_polarity) == int(predict):
        over_predict += 1
    # if int(over_polarity) == 1:
    #     POS += 1
    #     if int(aspect_polarity) == 1:
    #         POS1 += 1
    # elif int(over_polarity) == 0:
    #     NEU += 1
    #     if int(aspect_polarity) == 0:
    #         NEU1 += 1
    # elif int(over_polarity) == -1:
    #     NEG += 1
    #     if int(aspect_polarity) == -1:
    #         NEG1 += 1

oa_consistency = over_aspect / total_samples
op_consistency = over_predict/total_samples
print('over & aspect consistency rate:{}'.format(oa_consistency))
print('over & aspect consistency num:{}'.format(over_aspect))
print('over & predict consistency rate:{}'.format(op_consistency))
print('over & predict consistency:{}'.format(over_predict))
# print('POS:{}, NEU:{}, NEG:{}'.format(POS, NEU, NEG))
# print('POS1:{}, NEU1:{}, NEG1:{}'.format(POS1, NEU1, NEG1))
