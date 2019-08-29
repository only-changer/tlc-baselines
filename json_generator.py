import json

f = open('examples/roadnet.json','r')
roadnet = json.load(f)
intersections = [i for i in roadnet["intersections"] if not i["virtual"]]
print(len(intersections))
f.close()

phases = {}
for inter in intersections:
	phases[inter['id']] = []
	phase_info = inter["trafficLight"]["lightphases"]
	phase_id = 0
	for phase in phase_info:
		phases[inter['id']].append([phase_id, phase['time']])
		phase_id += 1

text = json.dumps(phases)
newf = open('examples/phases.json','w')
newf.write(text)
newf.close()

