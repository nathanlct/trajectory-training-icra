from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import sys


def plot_platoon_mpg(emissions_path, save_path):
	df = pd.read_csv(emissions_path)

	id_dict = {int(veh_id.split('_')[0]):{'veh_id': veh_id} for veh_id in df['id'].unique()}

	circles_nums = [id_num for id_num, veh_dict in id_dict.items() if 'av' in veh_dict['veh_id'] or 'sensor' in veh_dict['veh_id']]
	leader_nums = [circles_num-1 for circles_num in circles_nums]
	leader_nums.sort()
	local_nums = circles_nums + leader_nums
	local_nums.sort()
	local_ids = {id_dict[local_num]['veh_id']: local_num for local_num in local_nums}

	# df = df[df['id'].isin(local_ids.keys())]
	df = df.groupby('id').tail(1)
	color_dict = {
		'av': 'r',
		'sensor': 'cyan',
		'human': 'grey',
		'leader': 'grey',
	}
	df['color'] = df['id'].apply(lambda x: [v for k, v in color_dict.items() if k in x][0])
	df['id_num'] = df['id'].apply(lambda x: int(x.split('_')[0]))
	df.loc[~df['id_num'].isin(local_nums), 'avg_mpg'] = 0.0
	df = df.sort_values('id_num', ascending=False)

	av_nums = [id_num for id_num, veh_dict in id_dict.items() if 'av' in veh_dict['veh_id']]
	df['platoon_id'] = pd.cut(df['id_num'], [0] + av_nums + [100], right=False, include_lowest=True, labels=range(1, len(av_nums)+2))

	platoon_df = df[df['id'].isin(local_ids.keys())].groupby('platoon_id')[['total_miles', 'total_gallons']].sum()
	platoon_df['mpg'] = platoon_df['total_miles'] / platoon_df['total_gallons']

	platoon_df = platoon_df.sort_index(ascending=False)
	platoon_df['improvement_rel'] = 100 * (1 - (platoon_df['mpg'].shift(-1) / platoon_df['mpg']))
	platoon_df['improvement_abs'] = 100 * (1 - (platoon_df.loc[0, 'mpg'] / platoon_df['mpg']))

	plt.figure(figsize=(15,20))

	ax1 = plt.subplot(411)
	df.plot.bar(x='id_num', y='avg_mpg', color=df['color'], ax=ax1, edgecolor='k')
	max_mpg = df['avg_mpg'].max()
	plt.ylabel('Fuel Economy (mpg)', fontsize=20)
	plt.xlabel('Vehicles', fontsize=20)
	plt.title('Scenario 1', fontsize=24)

	ax1.add_patch(Rectangle((-0.5, 0), len(id_dict) - av_nums[-1] - 1, max_mpg * 1.1, edgecolor = 'g', fill=False, lw=2))
	plt.annotate('Platoon 5', (0, max_mpg * 1.2), fontsize=16)
	ax1.add_patch(Rectangle((len(id_dict) - av_nums[-1] - 0.5, 0), av_nums[3] - av_nums[2] - 1, max_mpg * 1.1, edgecolor = 'g', fill=False, lw=2))
	plt.annotate('Platoon 4', (len(id_dict) - av_nums[-1], max_mpg * 1.2), fontsize=16)
	ax1.add_patch(Rectangle((len(id_dict) - av_nums[2] - 0.5, 0), av_nums[2] - av_nums[1] - 1, max_mpg * 1.1, edgecolor = 'g', fill=False, lw=2))
	plt.annotate('Platoon 3', (len(id_dict) - av_nums[2], max_mpg * 1.2), fontsize=16)
	ax1.add_patch(Rectangle((len(id_dict) - av_nums[1] - 0.5, 0), av_nums[1] - av_nums[0] - 1, max_mpg * 1.1, edgecolor = 'g', fill=False, lw=2))
	plt.annotate('Platoon 2', (len(id_dict) - av_nums[1], max_mpg * 1.2), fontsize=16)
	ax1.add_patch(Rectangle((len(id_dict) - av_nums[0] - 0.5, 0), av_nums[0], max_mpg * 1.1, edgecolor = 'g', fill=False, lw=2))
	plt.annotate('Platoon 1', (len(id_dict) - av_nums[0], max_mpg * 1.2), fontsize=16)

	ax1.set_ylim(0, max_mpg*1.4)
	ax1.set_xlim(-1, len(id_dict))
	ax1.get_legend().remove()

	ax2 = plt.subplot(412)
	platoon_df['mpg'].plot.bar(ax=ax2)
	plt.ylabel('Fuel Economy (mpg)', fontsize=20)
	plt.xlabel('Platoons', fontsize=20)

	ax3 = plt.subplot(413)
	platoon_df.iloc[:4]['improvement_rel'].plot.bar(ax=ax3)
	plt.ylabel('Fuel Savings (%)\n(Relative to Prior Platoon)', fontsize=20)
	plt.xlabel('Platoons', fontsize=20)

	ax4 = plt.subplot(414)
	platoon_df.iloc[:4]['improvement_abs'].plot.bar(ax=ax4)
	plt.ylabel('Fuel Savings (%)\n(Relative to First Platoon)', fontsize=20)
	plt.xlabel('Platoons', fontsize=20)

	plt.tight_layout()
	plt.savefig(save_path)


if __name__ == '__main__':
	emissions_path, save_path = sys.argv[1], sys.argv[2]
	plot_platoon_mpg(emissions_path, save_path)
