import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import codecs
sys.stdout = codecs.getwriter("iso-8859-1")(sys.stdout, 'xmlcharrefreplace')

symbols = ['rs','b^','g^','c^','m^','ko','yo']

graph_metadatas = [
{
'filename' : 'run_times.csv',
'x-label' : 'Model',
'y-label' : 'Average Runtime [msec]',
'type' : 'bar-logy-no_x_label-yfloat',
#'type' : 'bar-no_x_label-yfloat',
'labels' : ['AIDs', 'LINUX', 'IMDB', 'REDDIT'],
'fig_size' : (15,3.8),
}
]

xs = []
ys = []
linegraph_id = 0
bargraph_id = 0
for graph_metadata in graph_metadatas:
	filename = graph_metadata['filename']
	x_label = graph_metadata['x-label']
	y_label = graph_metadata['y-label']
	graph_type = graph_metadata['type']
	ylim = graph_metadata['ylim'] if 'ylim' in graph_metadata.keys() else None # only works on line graphs
	fig_size = graph_metadata['fig_size'] if 'fig_size' in graph_metadata.keys() else None
	multiply_y_by = graph_metadata['multiply_y_by'] if 'multiply_y_by' in graph_metadata.keys() else None
	
	fp = open(filename)#, encoding='utf-8-sig')
	csv_reader = csv.reader(fp, delimiter=',')
	x_vals, y_vals = [], []
	for row in csv_reader:
		if len(row) == 2:
			x_val, y_val = row
			if 'yfloat' in graph_type:
				y_val = float(y_val)
		else:
			x_val, y_val = row[0], row[1:]
			if 'yfloat' in graph_type:
				y_val = [float(y_val_elt) for y_val_elt in y_val]
		if 'xfloat' in graph_type:
			x_val = float(x_val)
		if row[0] == 'end':
			break
		else:
			x_vals.append(x_val)
			y_vals.append(y_val)

	if 'line' in graph_type:
		plt.plot(x_vals,y_vals,symbols[linegraph_id],linestyle='-',fillstyle='none', linewidth=3.0, mew=3, ms=9)
		if ylim != None:
			ylim_bot, ylim_top = ylim
			plt.ylim(top=ylim_top)
			plt.ylim(bottom=ylim_bot)
		plt.grid('on')
		ax, fig = plt.gca(), plt.gcf()
		ax.yaxis.grid(False)
		ax.xaxis.grid(True)

		if 'logx' in graph_type:
			ax.set_xscale('log', basex=2)
		if 'xlog' in graph_type:
			ax.set_xscale('symlog', basex=2, linthreshx=x_vals[1])

		ax.tick_params(labelsize=22)
		plt.xlabel(x_label, fontsize=25)
		plt.ylabel(y_label, fontsize=25)
		plt.tight_layout()
		linegraph_id += 1

	elif 'bar' in graph_type:
		labels = graph_metadata['labels']
		hatches = ['///', '...', '', 'xxx', '+++']
		N = len(labels)
		if N != 1:
			y_procs = [[y_val[i] for y_val in y_vals] for i in range(N)]	
		else:
			y_procs = [[float(y_val) for y_val in y_vals]]
		x_procs = np.arange(len(x_vals))
		width = 0.6/N
		if N == 1:
			offset = x_procs - width/2
		elif N % 2 == 1:
			offset = x_procs - (N-1)/2 * width
		else:
			offset = x_procs - (N/2-0.5) * width
			
		plt.subplot(111)
		ax, fig = plt.gca(), plt.gcf()
		rects = [ax.bar(offset + i * width, y_procs[i], width, label=labels[i], hatch=hatches[i], color='w', edgecolor='k', zorder=10) for i in range(N)]
		plt.grid('on')
		ax.yaxis.grid(True)
		ax.xaxis.grid(False)

		if 'logy' in graph_type:
			pass
			ax.set_yscale('log', basey=10)
		x_vals = ['k↓' if 'k_down' == x_val else x_val for x_val in x_vals]
		#print(x_vals)
		ax.set_xticks(x_procs)
		ax.set_xticklabels(x_vals)
		ax.tick_params(labelsize=14)
		if 'no_x_label' not in graph_type:
			plt.xlabel(x_label, fontsize=16)
		plt.ylabel(y_label, fontsize=16)
		ax.legend([rect[0] for rect in rects], labels, fontsize=14)
		fig.tight_layout()
		bargraph_id +=1

	else:
		raise NotImplementedError

	if fig_size != None:
		fig.set_size_inches(fig_size[0], fig_size[1])
	plt.tight_layout()
	#plt.show()
	plt.savefig('fig_{}.png'.format(filename.split('.')[0]))
	plt.close()
	fp.close()
