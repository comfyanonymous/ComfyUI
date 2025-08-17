def filter(segs, labels):
    labels = set([label.strip() for label in labels])

    if 'all' in labels:
        return (segs, (segs[0], []), )
    else:
        res_segs = []
        remained_segs = []

        for x in segs[1]:
            if x.label in labels:
                res_segs.append(x)
            elif 'eyes' in labels and x.label in ['left_eye', 'right_eye']:
                res_segs.append(x)
            elif 'eyebrows' in labels and x.label in ['left_eyebrow', 'right_eyebrow']:
                res_segs.append(x)
            elif 'pupils' in labels and x.label in ['left_pupil', 'right_pupil']:
                res_segs.append(x)
            else:
                remained_segs.append(x)

    return ((segs[0], res_segs), (segs[0], remained_segs), )
