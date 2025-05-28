// paste assemblies/granola-ul/chr9-SymGatedGCN/{0,1,2,3,4}/assembly/0_quast/report.txt | awk 'NR==16;NR==17;NR==23;NR==41;NR==44;NR==45;NR==49;'

// # contigs                    41         # contigs                    39         # contigs                    40         # contigs                    43         # contigs                    39
// Largest contig               25621788   Largest contig               23001162   Largest contig               23079459   Largest contig               21696065   Largest contig               23612376
// NG50                         25621788   NG50                         23001162   NG50                         23079459   NG50                         21696065   NG50                         23612376
// Genome fraction (%)          92.309     Genome fraction (%)          91.687     Genome fraction (%)          94.325     Genome fraction (%)          92.598     Genome fraction (%)          92.515
// # mismatches per 100 kbp     6.54       # mismatches per 100 kbp     14.69      # mismatches per 100 kbp     9.00       # mismatches per 100 kbp     8.53       # mismatches per 100 kbp     10.50
// # indels per 100 kbp         3.88       # indels per 100 kbp         3.81       # indels per 100 kbp         4.31       # indels per 100 kbp         4.01       # indels per 100 kbp         4.44
// NGA50                        1337501    NGA50                        1855700    NGA50                        1014668    NGA50                        1415536    NGA50                        1711793

#let granola-ul-chr9-SymGatedGCN-contigs = (41, 39, 40, 43, 39)
#let granola-ul-chr9-SymGatedGCN-largest-contig = (25621788, 23001162, 23079459, 21696065, 23612376)
#let granola-ul-chr9-SymGatedGCN-ng50 = (25621788, 23001162, 23079459, 21696065, 23612376)
#let granola-ul-chr9-SymGatedGCN-genome-fraction = (92.309, 91.687, 94.325, 92.598, 92.515)
#let granola-ul-chr9-SymGatedGCN-mismatches = (6.54, 14.69, 9.00, 8.53, 10.50)
#let granola-ul-chr9-SymGatedGCN-indels = (3.88, 3.81, 4.31, 4.01, 4.44)
#let granola-ul-chr9-SymGatedGCN-nga50 = (1337501, 1855700, 1014668, 1415536, 1711793)

// paste assemblies/granola-ul/chr9-GAT/{0,1,2,3,4}/assembly/0_quast/report.txt | awk 'NR==16;NR==17;NR==23;NR==41;NR==44;NR==45;NR==49;'

// # contigs                    39         # contigs                    46         # contigs                    42         # contigs                    44         # contigs                    44
// Largest contig               16504614   Largest contig               27985002   Largest contig               22515885   Largest contig               18881596   Largest contig               28141679
// NG50                         13898584   NG50                         27985002   NG50                         22515885   NG50                         18881596   NG50                         28141679
// Genome fraction (%)          94.345     Genome fraction (%)          91.996     Genome fraction (%)          93.541     Genome fraction (%)          93.516     Genome fraction (%)          91.327
// # mismatches per 100 kbp     16.17      # mismatches per 100 kbp     12.73      # mismatches per 100 kbp     12.88      # mismatches per 100 kbp     7.82       # mismatches per 100 kbp     13.85
// # indels per 100 kbp         4.41       # indels per 100 kbp         3.83       # indels per 100 kbp         3.73       # indels per 100 kbp         3.67       # indels per 100 kbp         3.70
// NGA50                        1741099    NGA50                        1799244    NGA50                        1172601    NGA50                        1413399    NGA50                        1714844

#let granola-ul-chr9-GAT-contigs = (39, 46, 42, 44, 44)
#let granola-ul-chr9-GAT-largest-contig = (16504614, 27985002, 22515885, 18881596, 28141679)
#let granola-ul-chr9-GAT-ng50 = (13898584, 27985002, 22515885, 18881596, 28141679)
#let granola-ul-chr9-GAT-genome-fraction = (94.345, 91.996, 93.541, 93.516, 91.327)
#let granola-ul-chr9-GAT-mismatches = (16.17, 12.73, 12.88, 7.82, 13.85)
#let granola-ul-chr9-GAT-indels = (4.41, 3.83, 3.73, 3.67, 3.70)
#let granola-ul-chr9-GAT-nga50 = (1741099, 1799244, 1172601, 1413399, 1714844)


// paste assemblies/granola-ul/chr9-SymGAT/{0,1,2,3,4}/assembly/0_quast/report.txt | awk 'NR==16;NR==17;NR==23;NR==41;NR==44;NR==45;NR==49;'

// # contigs                    38         # contigs                    43         # contigs                    39         # contigs                    35         # contigs                    41
// Largest contig               22343585   Largest contig               27971199   Largest contig               26181314   Largest contig               15902442   Largest contig               26038775
// NG50                         22343585   NG50                         27971199   NG50                         26181314   NG50                         7372063    NG50                         26038775
// Genome fraction (%)          92.498     Genome fraction (%)          91.935     Genome fraction (%)          91.430     Genome fraction (%)          93.078     Genome fraction (%)          92.145
// # mismatches per 100 kbp     6.44       # mismatches per 100 kbp     8.31       # mismatches per 100 kbp     14.48      # mismatches per 100 kbp     10.78      # mismatches per 100 kbp     9.29
// # indels per 100 kbp         3.88       # indels per 100 kbp         3.89       # indels per 100 kbp         3.92       # indels per 100 kbp         3.57       # indels per 100 kbp         4.11
// NGA50                        1845270    NGA50                        1052609    NGA50                        2165525    NGA50                        2336981    NGA50                        1970349

#let granola-ul-chr9-SymGAT-contigs = (38, 43, 39, 35, 41)
#let granola-ul-chr9-SymGAT-largest-contig = (22343585, 27971199, 26181314, 15902442, 26038775)
#let granola-ul-chr9-SymGAT-ng50 = (22343585, 27971199, 26181314, 7372063, 26038775)
#let granola-ul-chr9-SymGAT-genome-fraction = (92.498, 91.935, 91.430, 93.078, 92.145)
#let granola-ul-chr9-SymGAT-mismatches = (6.44, 8.31, 14.48, 10.78, 9.29)
#let granola-ul-chr9-SymGAT-indels = (3.88, 3.89, 3.92, 3.57, 4.11)
#let granola-ul-chr9-SymGAT-nga50 = (1845270, 1052609, 2165525, 2336981, 1970349)