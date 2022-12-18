from collections import Counter

def slot_mapping(slot_value_gt, slot_value_hyp):
    sort_gt = sorted(slot_value_gt.items(), key=lambda x: len(x[1]), reverse=True)
    value2slot_hyp = {}
    #print('hyp...........')
    for slot, value in slot_value_hyp.items():
        #print('slot: ', slot)
        #print('value: ', Counter(value))
        for v in value:
            if v not in value2slot_hyp.keys():
                value2slot_hyp[v] = []
            value2slot_hyp[v].append(slot)

    #print('gt...........')
    slot_map = {}
    candidate_id = list(slot_value_hyp.keys())
    #print( 'candidate_id: ', candidate_id  )
    for slot, value in sort_gt:
        #print('slot: ', slot)
        #print('value: ', Counter(value))
        #print('len_value: ', len(value))

        slot_id_in_hyp = []
        for v in value:
            try:
                slot_id_in_hyp.extend(value2slot_hyp[v])
            except:
                pass
        counter = Counter(slot_id_in_hyp)
        
        #print('len(slot_id_in_hyp): ', len(slot_id_in_hyp))
        #print('Counter: ', counter)
        #print ([k for k,v in counter.items()])
        for k, v in counter.items():
            if k in candidate_id:
                slot_map[k] = slot
                candidate_id.remove(k)
                break
    return slot_map
    #print('slot_map: ', slot_map)