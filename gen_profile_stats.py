import statistics

with open('profiles/sample_profile_ont_ul.fastq', 'r') as file:
        new_data = []
        for line in file:
            data = line.rstrip()
            data_len = len(data)
            new_data.append(line)
        with open('profiles/sample_profile_ont_ul.stats', 'w') as dest:
            dest.write(f'num\t    {str(len(new_data))}\n')
            dest.write(f'len_total\t    {str(sum([len(read) for read in new_data]))}\n')
            dest.write(f'len_min\t    {str(min([len(read) for read in new_data]))}\n')
            dest.write(f'len_max\t    {str(max([len(read) for read in new_data]))}\n')
            dest.write(f'len_mean\t    {str(statistics.mean([len(read) for read in new_data]))}\n')
            dest.write(f'len_sd\t    {str(statistics.stdev([len(read) for read in new_data]))}\n')
            dest.write(f'accuracy_mean\t    0.998437\n')
            dest.write(f'accuracy_sd\t    0.001987\n')

