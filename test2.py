r = []
mins[0]
for loc in model.locs:
    r.append(np.nanmean([mins[0][i] for i in loc]))

