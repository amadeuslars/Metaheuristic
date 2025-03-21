max_values = [8959782.0, 18387821.0, 46770347.0,  76627567.0, 170784643.0]
best_values = [2399059.0, 6565898.0, 15784272.0, 24773905.0, 67230492.0]
imporvements = []
for i in range(len(max_values)):
    imporvements.append((max_values[i] - best_values[i])/max_values[i])

print(imporvements)