// paste assemblies/base/chr9-SymGatedGCN/{0,1,2,3,4}/assembly/0_quast/report.txt | awk 'NR==16;NR==17;NR==23;NR==41;NR==44;NR==45;NR==49;'

// # contigs                    47         # contigs                    46         # contigs                    44         # contigs                    44         # contigs                    45
// Largest contig               24500137   Largest contig               26211834   Largest contig               28012991   Largest contig               28124744   Largest contig               22465876
// NG50                         24500137   NG50                         26211834   NG50                         28012991   NG50                         28124744   NG50                         22465876
// Genome fraction (%)          93.581     Genome fraction (%)          90.933     Genome fraction (%)          93.870     Genome fraction (%)          92.531     Genome fraction (%)          92.717
// # mismatches per 100 kbp     13.23      # mismatches per 100 kbp     6.80       # mismatches per 100 kbp     13.16      # mismatches per 100 kbp     11.90      # mismatches per 100 kbp     9.83
// # indels per 100 kbp         4.40       # indels per 100 kbp         4.08       # indels per 100 kbp         4.11       # indels per 100 kbp         3.96       # indels per 100 kbp         3.96
// NGA50                        2339711    NGA50                        2101744    NGA50                        1218281    NGA50                        1925231    NGA50                        2101742

#let base-chr9-SymGatedGCN-contigs = (47, 46, 44, 44, 45)
#let base-chr9-SymGatedGCN-largest-contig = (24500137, 26211834, 28012991, 28124744, 22465876)
#let base-chr9-SymGatedGCN-ng50 = (24500137, 26211834, 28012991, 28124744, 22465876)
#let base-chr9-SymGatedGCN-genome-fraction = (93.581, 90.933, 93.870, 92.531, 92.717)
#let base-chr9-SymGatedGCN-mismatches = (13.23, 6.80, 13.16, 11.90, 9.83)
#let base-chr9-SymGatedGCN-indels = (4.40, 4.08, 4.11, 3.96, 3.96)
#let base-chr9-SymGatedGCN-nga50 = (2339711, 2101744, 1218281, 1925231, 2101742)


// paste assemblies/base/chr9-GAT/{0,1,2,3,4}/assembly/0_quast/report.txt | awk 'NR==16;NR==17;NR==23;NR==41;NR==44;NR==45;NR==49;'

// # contigs                    37         # contigs                    39         # contigs                    41         # contigs                    43         # contigs                    40
// Largest contig               28164802   Largest contig               28198545   Largest contig               28192511   Largest contig               28035673   Largest contig               22600046
// NG50                         28164802   NG50                         28198545   NG50                         28192511   NG50                         28035673   NG50                         22600046
// Genome fraction (%)          92.477     Genome fraction (%)          93.532     Genome fraction (%)          92.940     Genome fraction (%)          92.978     Genome fraction (%)          91.041
// # mismatches per 100 kbp     8.78       # mismatches per 100 kbp     10.74      # mismatches per 100 kbp     8.03       # mismatches per 100 kbp     10.58      # mismatches per 100 kbp     9.95
// # indels per 100 kbp         4.37       # indels per 100 kbp         4.70       # indels per 100 kbp         3.75       # indels per 100 kbp         4.81       # indels per 100 kbp         3.98
// NGA50                        2088276    NGA50                        2088287    NGA50                        2006689    NGA50                        1590215    NGA50                        2056043

#let base-chr9-GAT-contigs = (37, 39, 41, 43, 40)
#let base-chr9-GAT-largest-contig = (28164802, 28198545, 28192511, 28035673, 22600046)
#let base-chr9-GAT-ng50 = (28164802, 28198545, 28192511, 28035673, 22600046)
#let base-chr9-GAT-genome-fraction = (92.477, 93.532, 92.940, 92.978, 91.041)
#let base-chr9-GAT-mismatches = (8.78, 10.74, 8.03, 10.58, 9.95)
#let base-chr9-GAT-indels = (4.37, 4.70, 3.75, 4.81, 3.98)
#let base-chr9-GAT-nga50 = (2088276, 2088287, 2006689, 1590215, 2056043)


// paste assemblies/base/chr9-SymGAT/{0,1,2,3,4}/assembly/0_quast/report.txt | awk 'NR==16;NR==17;NR==23;NR==41;NR==44;NR==45;NR==49;'

// # contigs                    44         # contigs                    45         # contigs                    45         # contigs                    48         # contigs                    47
// Largest contig               16696291   Largest contig               26197650   Largest contig               22629441   Largest contig               27993242   Largest contig               26167785
// NG50                         11524522   NG50                         26197650   NG50                         22629441   NG50                         27993242   NG50                         26167785
// Genome fraction (%)          92.898     Genome fraction (%)          92.638     Genome fraction (%)          92.612     Genome fraction (%)          92.852     Genome fraction (%)          90.925
// # mismatches per 100 kbp     7.26       # mismatches per 100 kbp     7.96       # mismatches per 100 kbp     6.92       # mismatches per 100 kbp     13.98      # mismatches per 100 kbp     11.94
// # indels per 100 kbp         4.08       # indels per 100 kbp         4.49       # indels per 100 kbp         4.13       # indels per 100 kbp         4.16       # indels per 100 kbp         4.01
// NGA50                        2256347    NGA50                        2101743    NGA50                        2101743    NGA50                        2239916    NGA50                        1993522

#let base-chr9-SymGAT-contigs = (44, 45, 45, 48, 47)
#let base-chr9-SymGAT-largest-contig = (16696291, 26197650, 22629441, 27993242, 26167785)
#let base-chr9-SymGAT-ng50 = (11524522, 26197650, 22629441, 27993242, 26167785)
#let base-chr9-SymGAT-genome-fraction = (92.898, 92.638, 92.612, 92.852, 90.925)
#let base-chr9-SymGAT-mismatches = (7.26, 7.96, 6.92, 13.98, 11.94)
#let base-chr9-SymGAT-indels = (4.08, 4.49, 4.13, 4.16, 4.01)
#let base-chr9-SymGAT-nga50 = (2256347, 2101743, 2101743, 2239916, 1993522)

