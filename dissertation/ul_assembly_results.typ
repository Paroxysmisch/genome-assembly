// paste assemblies/ul/chr9-SymGatedGCN/{0,1,2,3,4}/assembly/0_quast/report.txt | awk 'NR==16;NR==17;NR==23;NR==41;NR==44;NR==45;NR==49;'

// # contigs                    39         # contigs                    37         # contigs                    39         # contigs                    40         # contigs                    42
// Largest contig               24374693   Largest contig               26106150   Largest contig               27971684   Largest contig               19165755   Largest contig               22399819
// NG50                         24374693   NG50                         26106150   NG50                         27971684   NG50                         19165755   NG50                         22399819
// Genome fraction (%)          93.995     Genome fraction (%)          95.075     Genome fraction (%)          91.845     Genome fraction (%)          94.639     Genome fraction (%)          92.400
// # mismatches per 100 kbp     6.49       # mismatches per 100 kbp     8.95       # mismatches per 100 kbp     7.10       # mismatches per 100 kbp     16.58      # mismatches per 100 kbp     14.42
// # indels per 100 kbp         3.69       # indels per 100 kbp         4.01       # indels per 100 kbp         3.71       # indels per 100 kbp         4.72       # indels per 100 kbp         5.26
// NGA50                        2275948    NGA50                        1335604    NGA50                        2030395    NGA50                        1766110    NGA50                        1766198

#let ul-chr9-SymGatedGCN-contigs = (39, 37, 39, 40, 42)
#let ul-chr9-SymGatedGCN-largest-contig = (24374693, 26106150, 27971684, 19165755, 22399819)
#let ul-chr9-SymGatedGCN-ng50 = (24374693, 26106150, 27971684, 19165755, 22399819)
#let ul-chr9-SymGatedGCN-genome-fraction = (93.995, 95.075, 91.845, 94.639, 92.400)
#let ul-chr9-SymGatedGCN-mismatches = (6.49, 8.95, 7.10, 16.58, 14.42)
#let ul-chr9-SymGatedGCN-indels = (3.69, 4.01, 3.71, 4.72, 5.26)
#let ul-chr9-SymGatedGCN-nga50 = (2275948, 1335604, 2030395, 1766110, 1766198)


// paste assemblies/ul/chr9-GAT/{0,1,2,3,4}/assembly/0_quast/report.txt | awk 'NR==16;NR==17;NR==23;NR==41;NR==44;NR==45;NR==49;'

// # contigs                    45         # contigs                    44         # contigs                    35         # contigs                    37         # contigs                    40
// Largest contig               24544817   Largest contig               28217966   Largest contig               23611403   Largest contig               27318198   Largest contig               28012990
// NG50                         24544817   NG50                         28217966   NG50                         23611403   NG50                         27318198   NG50                         28012990
// Genome fraction (%)          93.137     Genome fraction (%)          92.645     Genome fraction (%)          94.583     Genome fraction (%)          90.712     Genome fraction (%)          92.215
// # mismatches per 100 kbp     12.62      # mismatches per 100 kbp     9.99       # mismatches per 100 kbp     13.81      # mismatches per 100 kbp     9.51       # mismatches per 100 kbp     12.55
// # indels per 100 kbp         3.92       # indels per 100 kbp         4.20       # indels per 100 kbp         4.91       # indels per 100 kbp         5.20       # indels per 100 kbp         4.45
// NGA50                        2117084    NGA50                        902751     NGA50                        1337514    NGA50                        801904     NGA50                        1659535

#let ul-chr9-GAT-contigs = (45, 44, 35, 37, 40)
#let ul-chr9-GAT-largest-contig = (24544817, 28217966, 23611403, 27318198, 28012990)
#let ul-chr9-GAT-ng50 = (24544817, 28217966, 23611403, 27318198, 28012990)
#let ul-chr9-GAT-genome-fraction = (93.137, 92.645, 94.583, 90.712, 92.215)
#let ul-chr9-GAT-mismatches = (12.62, 9.99, 13.81, 9.51, 12.55)
#let ul-chr9-GAT-indels = (3.92, 4.20, 4.91, 5.20, 4.45)
#let ul-chr9-GAT-nga50 = (2117084, 902751, 1337514, 801904, 1659535)


// paste assemblies/ul/chr9-SymGAT/{0,1,2,3,4}/assembly/0_quast/report.txt | awk 'NR==16;NR==17;NR==23;NR==41;NR==44;NR==45;NR==49;'

// # contigs                    39         # contigs                    36         # contigs                    38         # contigs                    40         # contigs                    41
// Largest contig               24489212   Largest contig               23081010   Largest contig               24874058   Largest contig               24471774   Largest contig               22863833
// NG50                         24489212   NG50                         23081010   NG50                         24874058   NG50                         24471774   NG50                         22863833
// Genome fraction (%)          92.440     Genome fraction (%)          93.467     Genome fraction (%)          93.857     Genome fraction (%)          92.915     Genome fraction (%)          93.125
// # mismatches per 100 kbp     8.02       # mismatches per 100 kbp     8.66       # mismatches per 100 kbp     7.46       # mismatches per 100 kbp     12.21      # mismatches per 100 kbp     9.98
// # indels per 100 kbp         4.41       # indels per 100 kbp         3.42       # indels per 100 kbp         4.02       # indels per 100 kbp         4.91       # indels per 100 kbp         4.53
// NGA50                        966889     NGA50                        2101586    NGA50                        1925246    NGA50                        1925246    NGA50                        1997104

#let ul-chr9-SymGAT-contigs = (39, 36, 38, 40, 1)
#let ul-chr9-SymGAT-largest-contig = (24489212, 23081010, 24874058, 24471774, 22863833)
#let ul-chr9-SymGAT-ng50 = (24489212, 23081010, 24874058, 24471774, 22863833)
#let ul-chr9-SymGAT-genome-fraction = (92.440, 93.467, 93.857, 92.915, 93.125)
#let ul-chr9-SymGAT-mismatches = (8.02, 8.66, 7.46, 12.21, 9.98)
#let ul-chr9-SymGAT-indels = (4.41, 3.42, 4.02, 4.91, 4.53)
#let ul-chr9-SymGAT-nga50 = (966889, 2101586, 1925246, 1925246, 1997104)