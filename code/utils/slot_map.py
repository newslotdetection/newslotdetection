
from collections import Counter
import json
with open('slot_value_gt.json', 'r') as f:
    slot_value_gt = json.load(f)
with open('slot_value_hyp.json', 'r') as f:
    slot_value_hyp = json.load(f)

def slot_mapping(slot_value_gt, slot_value_hyp):
    sort_gt = sorted(slot_value_gt.items(), key=lambda x: len(x[1]), reverse=True)
    value2slot_hyp = {}
    print('hyp...........')
    slot_in_hyp = []
    for slot, value in slot_value_hyp.items():
        slot_in_hyp.append(slot)
        #print('slot: ', slot)
        #print('value: ', Counter(value))
        for v in value:
            if v not in value2slot_hyp.keys():
                value2slot_hyp[v] = []
            value2slot_hyp[v].append(slot)

    print('gt...........')
    slot_map = {}
    candidate_id = [str(i) for i in range(len(slot_value_gt))]
    print( 'candidate_id: ', candidate_id  )
    for slot, value in sort_gt:
        print('start map...................')
        print('slot: ', slot)
        print('value: ', Counter(value))
        print('len_value: ', len(value))

        slot_id_in_hyp = []
        for v in value:
            try:
                slot_id_in_hyp.extend(value2slot_hyp[v])
            except:
                pass
        counter = Counter(slot_id_in_hyp)
        
        print('len(slot_id_in_hyp): ', len(slot_id_in_hyp))
        print('Counter: ', counter)
        print ([k for k,v in counter.most_common()])
        for k, v in counter.most_common():
            if k in candidate_id:
                slot_map[k] = slot
                print('slot_map: ', slot_map)
                candidate_id.remove(k)
                break
            else:
                print('len(candidate_id)',len(candidate_id))
                print('candidate_id',candidate_id)
                if len(candidate_id)==2:
                    slot_map[candidate_id[0]] = slot
                    candidate_id.remove(candidate_id[0])
                print('slot_map: ', slot_map)
                
    print(candidate_id)
    print('slot_map: ', slot_map)
    print('slot_in_hyp: ', slot_in_hyp)
    print('sort_gt: ',slot_value_gt.keys()) 


    #for slot in slot_in_hyp:

    #    if not slot in slot_map.keys():
    #        slot_map[slot]="noslot"
            
    return slot_map

slot_map=    slot_mapping(slot_value_gt, slot_value_hyp)