#import "@preview/glossarium:0.5.6": make-glossary, register-glossary, print-glossary, gls, glspl
#import "@preview/wordometer:0.1.4": word-count, total-words
#import "@preview/algorithmic:1.0.0"
#import algorithmic: algorithm, algorithm-figure, style-algorithm
#show: style-algorithm
#import "@preview/subpar:0.2.2"
#import "@preview/oxifmt:0.3.0": strfmt
#import "base_assembly_results.typ": *
#import "ul_assembly_results.typ": *
#import "granola_assembly_results.typ": *

#set page(margin: ("top": 20mm, "bottom": 20mm, "left": 25mm, "right": 25mm))
#set text(size: 12pt, font: "New Computer Modern")
#show math.equation: set text(font: "New Computer Modern Math")
#set heading(numbering: "1")
#set page(background: [#v(30%) #image("graphics/chr19.png", width: 75%, height: 45%, fit: "contain")])

#align(right)[
  #text(size: 1.5em)[Yash Shah]
]

#v(5%)

#text(size: 2.5em)[Graph Neural Networks for #linebreak() Accelerated Genome Assembly]
#line(length: 100%)
#v(1.5em)
#text(size: 1.5em)[
  #set par(spacing: 0.75em)
  Computer Science Tripos, Part III

  Gonville & Caius College

  June 2025
]

#align(bottom)[
  #grid(
    columns: 2,
    align: horizon,
    [#image("graphics/ucam-logo-colour-preferred.png", width: 4.5cm)],
    [#set align(right); Submitted in partial fulfillment of the requirements for the Computer Science Tripos, Part III]
  )
   
]

#pagebreak()
#set par(justify: true)
#let title(content) = [#v(8em) #text(size: 2em)[#content]]

#counter(page).update(1)
#set page(numbering: "I", background: none)

Total page count: #context counter(page).final().first()

Main chapters (excluding front-matter, references, and appendix): #context[#let start = locate(<start-main-body>).page(); #let end = locate(<end-main-body>).page(); #let diff = end - start + 1; #diff pages (pp 1--#diff)]

Main chapters word count: #total-words

Methodology used to generate that word count:
```typst
#import "@preview/wordometer:0.1.4": word-count, total-words

// Front-page, declaration, ...
Main chapters word count: #total-words
// ...

// Word count from main body onwards
#show: word-count.with(exclude: (<no-wc>))

// Main body ...

// Appendix
#[...] <no-wc>
```

#title[Declaration]

I, Yash Shah of Gonville & Caius College, being a candidate for Part III of the Computer Science Tripos, hereby declare that this report and the work described in it are my own work, unaided except as may be specified below, and that the report does not contain material that has already been used to any substantial extent for a comparable purpose. In preparation of this report, I adhered to the Department of Computer Science and Technology AI Policy. I am content for my report to be made available to the students and staff of the University.

Signed [signature]

Date [date]

#show: make-glossary
#import "glossary.typ": entry-list
#register-glossary(entry-list)


#pagebreak()

#title[Abstract]

Genome assembly is a cornerstone for computation biology, with an accurate reconstruction of an organism's genome being crucial to understanding its biology and evolution. This dissertation explores methods to improve the layout phase in the Overlap-Layout-Consensus genome assembly algorithm with the application of Graph Neural Networks. Conventional methods to clean and simplify overlap graphs utilize a collection of algorithms and heuristics that are insufficient to fully resolve all graph artifacts, ultimately relying on labor-intensive manual graph inspection. Graph Neural Networks have recently been shown to be a promising replacement to the traditional heuristic-oriented approach, helping improve assembly quality with increased contiguity and genome coverage. This project seeks to advance this neural assembly paradigm by exploring more advanced Graph Neural Network architectures, investigating graph-adaptive normalization, as well as automated feature extraction from raw sequencing data with the Mamba Selective State Space Model. Additionally, the advent of ultra-long sequencing technology offers new opportunities for resolving particularly complex genomic regions, such as long tandem repeats. This project integrates ultra-long reads with contemporary sequencing technology, yielding improved genome coverage, whilst maintaining assembly quality. Lastly, a proof of concept of an end-to-end neural assembly paradigm, where the neural network is not bounded to merely augmenting predefined assembly stages is also presented. This removes the constraints and biases imposed by the Overlap-Layout-Consensus framework. 

#pagebreak()

#title[Acknowledgements]

I would like to thank my supervisors Dobrik Georgiev, Lovro Vrček, Martin Schmitz, and Pietro Liò for their immense support and advice throughout this project. Without them, this project would not have been possible.

I would also like to thank my Directors of Studies Timothy Jones and Russell Moore for supporting me throughout my degree.

#pagebreak()

// #set heading(supplement: [Chapter])

#show heading.where(level: 1): it => [
  #set text(weight: "regular", size: 12pt)
  #v(8em)
  #text(size: 2em)[#it.body.text]
  #v(2em)
]

#show outline.entry.where(
  level: 1
): set block(above: 2em)

#outline()

#pagebreak()

#title[Acronyms]


// #print-glossary(
//  entry-list
// )

#let title-case(string) = {
  return string.replace(
    regex("^[[:alpha:]]+('[[:alpha:]]+)?"),
    word => upper(word.text.first()) + word.text.slice(1),
  )
}


// #[
// #set table(
//   fill: (x, y) => if calc.rem(y, 2) == 1 {
//       gray.lighten(65%)
//     }
// )
  
// #print-glossary(
//   user-print-glossary: (entries, groups, ..) => {
//     table(
//       columns: (0.4fr, 1fr),
//       stroke: 0pt,
//       row-gutter: 0.25em,
//       ..for group in groups {
//         (
//           table.cell(group, colspan: 2),
//           ..for entry in entries.filter(x => x.group == group) {
//             (
//               text(weight: "bold", style: "italic")[#entry.short],
//               // entry.long
//               title-case(entry.long),
//             )
//           }
//         )
//       }
//     )
//   },
//   entry-list
// )]

#print-glossary(
 entry-list,
 disable-back-references: true,
)

#show heading.where(level: 1): it => [
  #set text(weight: "regular", size: 12pt)
  #v(8em)
  // #it.supplement #context counter(heading).display("1")
  Chapter #context counter(heading).display("1")
  #linebreak()
  #text(size: 2em)[#it.body.text]
  #v(2em)
]

#set page(numbering: "1")
#set math.equation(numbering: "(1)")
#show: word-count.with(exclude: (<no-wc>))

#set heading(numbering: "1.")
#set place(float: true)
#show figure.caption: set text(size: 0.9em)
#show figure.caption: set align(left)
#let sub-caption-styling = (num, it) => [#set align(center); #num #it.body #v(0.5em)]
#show smallcaps: set text(font: "New Computer Modern")

#show figure.where(
  kind: table
): set figure.caption(position: top)

// #show link: it => text(style: "italic")[#it]

#counter(page).update(1)
<start-main-body>
= Introduction
In this chapter, we motivate the genome assembly problem, introduce the overlap graph simplification task, and provide some related work around neural genome assembly. We then state the aims and key contributions made by this project.

== Motivation
Genome assembly has remained a central subject in computational biology for the past four decades @t2t-genome-assembly, as accurate reconstruction of an organism’s genome is essential in understanding its biology and evolution @t2t-evolution. By enabling researchers to map and analyze an organism's @dna, including genes, regulatory elements, and non-coding regions (segments that do not directly encode proteins), we gain insight into the organism's traits, development, and overall function. Comparative analyses of genome assemblies across species also sheds light on evolutionary relationships, and has applications to fields such as genome-assisted breeding of more resistant crops @t2t-unlocks-organization-function.

In addition, assembling a large number of genomes from the same species allows scientists to study the role of genetic variation in health and disease @t2t-human-genome-implications, revealing factors that contribute to susceptibility or resistance to various conditions. This is increasingly important as we move into the realm of targeted healthcare, such as personalized drugs that are tailored to an individual by utilizing their unique genetic information, providing more effective treatment. 

A #gls("t2t", display: "telomere-to-telomere") assembly is essential in gaining complete insight into an organism's genome. A @t2t sequence represents a complete, continuous genome without gaps or fragmentation @t2t-genome-assembly. Remarkably, achieving such assemblies has only become feasible in recent years with the advent of long-read sequencing technologies. Although the Human Genome Project concluded in 2003 @hgp-facts, the reference genome produced contained several missing regions that were challenging to assemble, with the first @t2t sequencing of human DNA only recently achieved in 2021 @t2t-chm13.

In this project, we demonstrate how machine learning is an effective tool in improving de novo (without relying on a pre-existing reference genome) @t2t assembly, by increasing accuracy particularly in challenging genomic regions.

== A short history of genome assembly <sec:existing_methods>
Hierarchical sequencing and @wgs have been the two predominant assembly strategies @t2t-genome-assembly. Hierarchical sequencing involves cloning, sequencing, and assembly of tiled genomic fragments that are aligned against a physical or genetic genome map, with the human reference genome GRCh38 being primarily constructed with this method @grch38. 

Due to its high cost and labor-intensive nature, hierarchical sequencing has largely been replaced by @wgs, where the genome is randomly fragmented into individually sequenced smaller segments called reads @wgs1 @wgs2. These reads are then reassembled into a complete genome by identifying overlaps between them. Unlike hierarchical sequencing, @wgs must consider overlaps between reads spanning the entire genome, not just localized regions, which significantly increases computational complexity.

A fundamental part of @wgs is the creation of the overlap graph @overlapgraphsdetailed @lovro @overlapgraph, in which each vertex is a read. There exists a directed edge between the vertex of read $A$ and of read $B$ if the suffix of $A$ can be aligned to (i.e. overlaps with) the prefix $B$. The genome can then be reconstructed by finding a Hamiltonian path (a path that visits every vertex/read in the graph exactly once) through this overlap graph.

However, in practice, this overlap graph is not perfect. Due to the computational cost of exact overlap calculations and the inherent noise in sequencing technologies, overlaps are imprecise. Additional challenges include errors in base-calling (translating the electrical signals into a sequence of nucleotides: @nuc_a, @nuc_g, @nuc_c, @nuc_t) the raw read data, long repetitive regions in the genome, and other sequencing artifacts---all of which introduce spurious nodes and edges into the graph, which must be cleaned up @lovro.

#place(top + center)[#figure(image("graphics/artifacts.svg", width: 90%), caption: [Elementary artifact types encountered in overlap graphs. Each node in this example overlap graph represents a read, and the edges correspond to overlaps between those reads.]) <fig:common_artifacts>]

Existing methods for overlap graph simplification involve a collection of algorithms and heuristics to remove elementary artifacts such as bubbles (alternative paths between the same pair of nodes), dead-ends (paths branching off the main body), and transitive (shortcut) edges (@fig:common_artifacts) @overlapgraphsdetailed @lovro. It is vital to note that artifacts occurring in overlap graphs are not bound to only these three categories, and do not occur in isolation. Instead, they frequently have complex interactions that lead to challenging to resolve tangles (demonstrated in @full). 

Consequently, despite the utility of heuristic algorithms, these methods often struggle in complex genomic regions, where unique assembly solutions may not exist, resulting in either the omission of these complex regions in the final assembly completely, leading to a fragmented and incomplete result, or reliance on manual curation by human experts---an approach that is time-consuming, costly, and not scalable when processing thousands of genomes @lovro.

#place(top + center)[#subpar.grid(
  columns: (1fr, 1fr, 1fr),
  show-sub-caption: sub-caption-styling,
  figure(image("graphics/chr19_bubble.png"), caption: [
    Bubble
  ]), <bubble>,
  figure(image("graphics/chr21_tip.png"), caption: [
    Dead-end/Tip
  ]), <tip>,
  figure(image("graphics/chr19_tangle.png"), caption: [
    Transitive edges
  ]), <tangle>,
  caption: [The figures show common artifacts in overlap graphs generated from real read data, taken from human chromosome 19 (a, b) and 21 (c). Note that these are the very same overlap graphs that are successfully simplified and resolved by our work (visualized via Graphia @graphia).],
  label: <full>,
)]

== Related work
Deep learning has successfully addressed various genome assembly challenges, often resulting in state-of-the-art results across multiple sub-tasks. Neural networks have been applied to improve basecalling @neural-basecalling, which is the computational process by which the raw electrical signals output by the sequencing hardware are converted into the nucleotide sequences called reads. This improves the quality of the reads, reducing errors in the overlap graph subsequently generated. Transformer encoder models @transformer-paper have been used for sequence correction @neural-consensus, vastly enhancing assembly contiguity, gene completeness, base accuracy, and reducing false gene duplications and variant calling (identifying differences between the sequenced and reference genome) errors.

Prior work has used @gnn:pl to neurally executed common graph algorithms @neural-graph-algorithms, including the @tsp @tsp-gnn and Hamiltonian Path problem @inter-homo-gnn (which can be reduced to @tsp in polynomial time). We cannot simply use state-of-the-art neural Travelling Salesman solvers (like the use of Graph Transformers @tsp-graph-transformer) due to the size of overlap graphs (thousands of edges and millions of nodes) being much larger than those used during research for creating these models. Secondly, these models utilize node coordinate information that is not available in the overlap graph. Lastly, these models assume the input graphs are error-free---this is not true for overlap graphs.

The neural execution paradigm has been successful in simulating some deterministic overlap graph simplification algorithms, including transitive edge, dead-end/tip, and bubble removal @step-neural-genome-assembly, showcasing the potential of @gnn:pl for artifact resolution in overlap graphs (more precisely, the layout phase of the @olc algorithm, detailed in @sec:olc). Importantly, building on this, and the framework laid out by neural @tsp solving @tsp-gcn, @gnn:pl have been employed for direct artifact resolution in overlap graphs @lovro, without trying to replicate the result of a fixed existing algorithm, or heuristic. This was shown to improve assembly contiguity, and reduce mismatches and indels (erroneous insertions/deletions) in the reconstructed genome.

== Aims
While previous work @lovro has paved the way to replace the combination of algorithms and heuristics traditionally used during the layout phase of the @olc algorithm, this project aims to build on these advancements with three aims:

#let aim_1 = [Investigating the efficacy of more advanced @gnn layers, and normalization methods that hold promise for furthering performance.]

#let aim_2 = [Integrating ultra-long sequencing data into the @gnn\-based genome assembly pipeline, which may necessitate more advanced @gnn layers. This new data type offers new opportunities for resolving complex graph artifacts, particularly repeating regions in the genome that were previously difficult to address.]

#let aim_3 = [Exploring automated methods of feature extraction from raw nucleotide read data.]

#let aim_4 = [Evaluating the feasibility of an end-to-end neural approach to the genome assembly problem that goes beyond isolated improvements to various sub-tasks, such as layout.]

+ #aim_1

+ #aim_2

+ #aim_3

+ #aim_4

== Key contributions
The key contributions of this project are as follows:

+ Extension of the @gat architecture, called @gatedge, supporting updating of edge features, and their incorporation into message passing. The symmetry mechanism of @symgatedgcn introduced in prior work @lovro is then combined with @gatedge, to form a new architecture called SymGAT. (@sec:gat-edge and @sec:symgat-edge).

+ Evaluating the performance of these new architectures against @symgatedgcn. (@sec:performance_alt_gnn_layers).

+ Investigating the integration of ultra-long read data, by combining them into existing overlap graphs generated from long-reads only, and testing the performance of various @gnn architectures on this extended overlap graph. (@sec:explaining-ultra-long-data and @sec:integration-ultra-long-data).

+ Experimenting with the use of an alternative graph normalization scheme, specifically designed for @gnn:pl. (@sec:granola and @sec:granola-performance).

+ Incorporating much richer features by utilizing the raw nucleotide read data directly, resulting in the @symgatedgcn-mamba and @symgatedgcn-mambaedge models. (@sec:symgatedgcn-mamba, @sec:symgatedgcn-mamba-edge, and @sec:mamba-potential-feature-extraction).

+ Creating a proof-of-concept for purely neural genome assembly that does not rely on overlap graphs, or the @olc algorithm.

// Motivation
//   - Applications
//     - Could talk about in the future being able to integrate other types of read data
  
//   - Existing methods
//     - Little work done on using neural networks
//     - Talk about the heuristics that are used possibly

// Outline
//   - Aims
//   - Key contributions

#pagebreak()

= Background
This chapter provides biological background regarding: the genome assembly algorithm; limitations of existing sequencing technology; introduction to ultra-long reads, and their integration with the contemporary long-read datatype. This is followed by a primer on @gdl @gdl-book, @gnn:pl, and the Mamba Selective State Space Model @mamba.

== Overlap-Layout-Consensus <sec:olc>//https://bio.libretexts.org/Bookshelves/Computational_Biology/Book%3A_Computational_Biology_-_Genomes_Networks_and_Evolution_(Kellis_et_al.)/05%3A_Genome_Assembly_and_Whole-Genome_Alignment/5.02%3A_Genome_Assembly_I-_Overlap-Layout-Consensus_Approach
The fundamental problem in genome sequencing is that no current technology can read continuously from one end of the genome to the other @t2t-genome-assembly. Instead, sequencing technologies only produce relatively short contiguous fragments called reads. Most chromosomes are $>10$ @mb long, and can be up to $1$ @gb long @t2t-genome-assembly, while even current long-read sequencing technologies only produce accurate reads up to a few $100$s of @kb:pl @nanopore-ul. Thus assembling the genome requires an algorithm to combine these shorter reads. @olc @olc-algorithm is the predominant approach for genome assembly with long reads. In this section, we discuss the three phases of @olc in more detail.

#let kmer = [$k$-mer]
#let kmers = [$k$-mers]

#place(bottom + center)[#subpar.grid(
    columns: 2,
    gutter: 4em,
    show-sub-caption: sub-caption-styling,
    figure(image("graphics/overlap.svg"), caption: [Overlap between two reads]), <fig:overlap>,
    figure(image("graphics/kmer.svg"), caption: [All 3-mers present in a read]), <fig:kmer>,
    caption: [(a) shows the maximal overlapping region between a pair of reads, where the alignment is found using the Needleman-Wunsch dynamic programming algorithm. (b) shows all #kmers present in an example read, where $k=3$.]
)]


=== Overlap
The first step is identifying overlapping reads. Read $A$ overlaps with read $B$ if the suffix of $A$ matches the prefix of $B$ (shown in @fig:overlap). While the Needleman-Wunsch dynamic programming algorithm @needleman-wunsch can be used to find overlaps through pairwise alignments of reads, its $cal(O)(n^2)$ (where $n$ is the nucleotide length of the longer read) complexity for each pair of reads makes it infeasible for genome assembly involving millions or billions of read pairs. Moreover, most read pairs do not overlap, making exhaustive pairwise alignment highly inefficient.


An alternative approach, taken by the pairwise alignment software minimap2 @minimap2, is similar to the BLAST @blast-algorithm algorithm, leveraging $k$-mers---unique $k$-length substrings (example #kmers for a read are displayed in @fig:kmer) acting as seeds for identifying overlaps @olc-book. The algorithm extracts all #kmers from the reads and locates positions where multiple reads share common #kmers. An approximate similarity score is computed depending on the multiplicity and location of matching #kmers.

Next, read pairs (matches) falling under some threshold of similarity, say $95%$, are discarded. The full alignment need only be calculated for these remaining matching reads. The matches do not need to be identical, allowing tolerance for sequencing errors (and heterozygosity for diploid(/polyploid) organisms (like humans) where there may be two variants of an allele with one from each parent at polymorphic sites in the genome).

This overlap information is used to construct the overlap graph in which each vertex is a read and there exists a directed edge between the vertex of read $A$ and of read $B$ if they overlap (@fig:layout_phase).

=== Layout
#place(top + center)[#subpar.grid(
  columns: (1fr, 0.9fr, 1fr),
  align: top,
  gutter: 2em,
  show-sub-caption: sub-caption-styling,
  figure(image("graphics/layout_phase.svg", height: 8cm, fit: "contain"), caption: [Constructing the Overlap Graph]), <fig:layout_phase>,
  figure(image("graphics/sequencing_errors.svg", height: 8cm, fit: "contain"), caption: [Errors manifesting in the Overlap Graph]), <fig:sequencing_errors>,
  figure(image("graphics/resolving_repeats.svg", height: 8cm, fit: "contain"), caption: [Assembling tandem #linebreak() repeat regions]), <fig:resolving_repeats>,
  caption: [All figures show the reference genome to illustrate the relative genomic positions of the reads. (a) demonstrates the construction of the overlap graph from reads. Note that there are transitive edges in the resulting graph despite the presence of sequencing errors. Unitigging refers to the process of identifying a high-confidence contig---a contiguous sequence of DNA. (b) shows how artifacts manifest in the overlap graph. A sequencing error in Read 2 leads to the formation of a tip. #text(fill: green.darken(20%))[Green] regions in the reference genome correspond to segmental duplications that cannot be distinguished. Thus, the start of Read 4 appears identical to that of Read 1, leading to the creation of an erroneous transitive edge. (c) shows how a tandem repeat region in the genome (in #text(fill: red.lighten(20%))[pink]) can be resolved by using mutations in the @dna to differentiate different positions in the repetitive region. This requires exact matches to avoid addition of erroneous edges, necessitating high accuracy reads. Figures adapted from #cite(<t2t-genome-assembly>, form: "prose").]
)]

In a perfect overlap graph, free from artifacts, the genome can be reconstructed by finding a Hamiltonian path (a path that visits every vertex/read in the graph exactly once) @lovro. Contemporary assemblers first simplify the overlap graph by removing spurious vertices and edges (such as bubbles, dead-ends, and transitive edges) @minimap-miniasm @raven, aiming to simplify the graph into a chain. @fig:sequencing_errors shows how some of these errors form. However, as previously mentioned in @sec:existing_methods, this simplification is often incomplete, or infeasible.

This project targets this layout phase with the use of @gnn:pl @gnn-survey. The @gnn takes a partially simplified overlap graph as input, and predicts a probability of each edge belonging to the Hamiltonian path corresponding to the genome.



=== Consensus
#place(top + center)[#figure(
  image("graphics/consensus.svg"),
  caption: [The consensus phase is responsible for the correction of per-base errors within contigs identified by the layout phase. Nucleotides in #text(fill: red)[red] highlight differences compared to the majority at that position.]
) <fig:consensus>]

Each path found in the previous layout phase corresponds to a contig---a contiguous sequence of @dna constructed from the set of overlapping reads in the path, representing a region of the genome. However, recall that the reads are erroneous, and overlaps are inexact---consensus is the step to address per-base errors.

At any particular location within the contig, there may be multiple overlapping reads. These reads need to be aligned to the assembled contig at the base level, and then consensus on each nucleotide reached to produce the final contig (illustrated in @fig:consensus). There are multiple methods of achieving consensus post read alignment. For example, a simple majority vote can be taken for each nucleotide position, or a weighted scheme, using nucleotide quality scores as weights.

== Repeating regions and the need for accurate long-read technology <sec:need_long_reads>
// https://pmc.ncbi.nlm.nih.gov/articles/PMC1226196/
A sequence occurring multiple times in the genome is known as a repetitive sequence, or repeat, for short. The repeat structure, rather than the genome's length, is the predominant determinant for the difficulty of assembly @t2t-genome-assembly. Problematic repeating regions often consist of satellite repeats @satellite-repeats, which are short ($<100$ base pairs), almost identical @dna sequences that are repeated in tandem, as well as segmental duplications @segmental-duplications (also known as low-copy repeats) which are long ($1$--$400$ @kb) @dna regions that occur at multiple sites in the genome. A sufficiently long read may resolve such regions by bridging the repetitive segment and linking adjacent unique segments, however the lengths of these repetitive regions far exceed the lengths captured by any sequencing technology today. For instance, the pericentromeric region of human chromosome 1 contains 20 @mb of satellite repeats @satellite-repeats.

Fortunately, the repeat copies forming these extended repeat regions are inexact replicas due to the mutations acquired over time, and so do not share an identical repeat sequence spanning $> 10$ @kb @t2t-genome-assembly. Although highly accurate long reads cannot span the entire region, they can distinguish between these subtly differing inexact duplicates, with their length sufficing in bridging between the differences in the repeats. With help from the @olc algorithm detailed earlier, such repeating regions can be resolved and sequenced, as demonstrated in @fig:resolving_repeats. It is critical that such long reads have high accuracy, as errors in the reads are otherwise indistinct to the mutations we rely on to sequence such regions.

Recall that @t2t assembly is a gapless reconstruction of the genome. Although long-read technology was introduced in 2010, supported by the arrival of single-molecule sequencing @single-molecule-1 @single-molecule-2, their error rate ($~10%$) was too high to resolve complex genomic section @t2t-genome-assembly, like those exhibited by repeating regions. This resulted in fragmented, and incomplete assembly. Accurate long-read technology was only available since 2019 @long-reads, revolutionizing genome assembly, and making possible the first @t2t human genome assembly in 2021 @t2t-chm13.

== Sequencing technology <sec:sequencing-technology>
There are three key characteristics of sequencing reads that are often traded-off during genome assembly: length, accuracy, and evenness of representation. The ideal sequencing technology produces long, highly accurate reads, with uniform coverage across the genome---avoiding gaps in low-coverage regions, and conserving computational resources in over-represented areas. Contemporary efforts targeting de novo @t2t assembly focus on accurate long-read technology @long-reads-leading that produces contiguous sequences spanning #box[$>=10$ @kb] in length, with @pacbio and @ont being the two companies leading their development.

@pacbio's @hifi @hifi-and-ul read technology is the current core data type for high-quality genome assembly, due to its potential to generate reads spanning #box[$10$--$20$ @kb] in length, with an error rate $<0.5%$, replacing the previous continuous long-read solution that had an error rate $>10%$ @long-reads. Despite the success of long-read technology in achieving @t2t assemblies, the advent of ultra-long read technology is fast becoming a compelling additional data type to improve assembly reliability. @ont's @ul @hifi-and-ul sequencing technology is central in the generation of ultra-long read data, producing reads $>100$ @kb in length @ul-verko @double-graph @t2t-chm13, however with significantly lower accuracy ($90$--$95%$) than the @hifi solution.

Due to their much increased length, @ul is critical in helping resolve tangles, repeat sequences, and other artifacts that cannot be resolved with @hifi reads alone. At present, @ul reads are more expensive than @hifi data (in part as they require large amounts of input @dna), and so are not commonplace in current sequencing projects @t2t-genome-assembly. However, as the technology matures, they offer tremendous potential in improving the accuracy and scalability of @t2t assembly. Hence, in this project, we find exploring the incorporation of such ultra-long read data with the neural genome assembly paradigm incredibly valuable.

This project utilizes @pacbio's @hifi read technology for long-read data, as well as integrating @ont @ul for ultra-long read data. The next section discusses this integration in more detail.

== Integrating ultra-long data <sec:explaining-ultra-long-data>
As shown in @sec:need_long_reads, and demonstrated in @fig:resolving_repeats, long reads are crucial in helping resolve repeating regions and tangles in assembly graphs. Longer reads are critical in improving assembly quality, but only if their accuracy is maintained. Unfortunately, current ultra-long read technology's accuracy is not high enough to replace long reads as the primary data type. Hence, we have to incorporate them as additional information into existing long-read assembly workflows. @fig:ul_strategy shows that ultra-long data can help in resolving small assembly gaps and artifacts, e.g. a bubble can be simplified by the ultra-long read bridging the bubble region, and reinforcing a unique path through that tangle.

#place(top + center)[#figure(
  image("graphics/ul-strategy.svg"),
  caption: [Demonstration of ultra-long read data improving assembly quality. (A) The #text(fill: orange)[amber] reads correspond to @pacbio @hifi long reads, and the #text(fill: purple)[purple] reads reference @ont @ul reads. Note that the ultra-long reads contain more sequencing errors. (B) Error correction is applied to remove some sequencing errors. (C) Initial assembly graph generated from long-read data, with arrows representing sequences, and thin lines connecting them. Note the presence of artifacts such as bubbles and gaps. (D) By threading ultra-long reads through this assembly graph, artifacts can be resolved, and assembly gaps patched. Figure adapted from #cite(<t2t-genome-assembly>, form: "prose").]
) <fig:ul_strategy>]

A naive method of ultra-long read integration is to construct two assembly graphs--- one solely from long reads, and the other from ultra-long reads. Then, these assembly graphs could be combined. Alternatively, the ultra-long reads could be treated simply as additional read data that is used to construct the assembly graph. Unfortunately, neither of these approaches would lead to a high quality assembly due to numerous issues.

Firstly, identifying correct overlaps among ultra-long reads is particularly challenging due to their higher error rate @double-graph. Secondly, the @ont @ul technology in particular suffers from an increased frequency of recurrent sequence errors, making overlap identification even more problematic in complex genomic regions @double-graph. Lastly, computing all-to-all pairwise overlaps is the predominant computational bottleneck in long-read assembly. Ultra-long reads increase these computational demands even further.

#place(top + center)[#figure(
  image("graphics/hifiasm_ul.svg"),
  caption: [Double graph framework used in `Hifiasm (UL)` to integrate @ont @ul reads with long-read information. (A) A string graph from only @pacbio @hifi reads is constructed, and ultra-long reads aligned to these long reads. (B) Ultra-long reads are translated from base-space to integer-space. (C) Overlaps between ultra-long reads are calculated in integer space, and an integer graph created. Contigs are then found in this integer graph. (D) The ultra-long contigs are integrated into the @hifi string graph. (E) Additional graph cleaning can be performed using ultra-long data. For example, the number of ultra-long reads supporting each edge can be tracked. In the case of the bubble, no ultra-long reads supported the alternative path, hence resolving the bubble. Figure adapted from #cite(<double-graph>, form: "prose").]
) <fig:hifiasm_ul>]

An alternative approach, employed by `Hifiasm (UL)` @double-graph, which is the assembler utilized by this project, is the double graph framework (illustrated in @fig:hifiasm_ul) that exploits all information contained in both sets of reads. The @pacbio @hifi long-reads are initially used to create a string graph---an assembly graph preserving read information. Next, the @ont @ul reads are aligned to these @pacbio @hifi reads. This alignment information is then used to map the ultra-long reads from base-space into integer space---instead of each ultra-long read being a sequence of nucleotides, it is now a sequence of integer node identifiers from the @hifi string graph.

Each ultra-long read in integer space is only $10$s of node identifiers long, instead of $100$s of @kb, allowing for inexpensive all-to-all overlap calculation that is also accurate---the underlying nucleotide information is from the much more accurate @hifi reads. With ultra-long overlaps calculated, an ultra-long integer (overlap) graph can be constructed, that is then used to extract ultra-long integer contigs. These ultra-long contigs can then be incorporated into the original @hifi string graph. During this integration, the additional information provided by the ultra-long contigs can help clean the original @hifi assembly (as shown in @fig:hifiasm_ul (D)).

While the integration of ultra-long data may help eliminate some overlap graph artifacts, it introduces new erroneous nodes and edges too. This is a result of issues such as: ultra-long reads having a much higher error rate; reliance on imperfect alignment with long reads, and erroneous integer sequence overlap calculation. Ultra-long reads are poised to be a valuable data type moving forward, and so it is compelling to evaluate their utility with neural genome assembly.



== Geometric Deep Learning
@gdl @gdl-book is a framework leveraging the geometry in data, through groups, representations, and principles of invariance and equivariance, to learn more effective machine learning models. 

Central to @gdl are symmetries---transformations that leave an object unchanged. In the context of machine learning, relevant symmetries can arise in various forms: symmetries of the input data (e.g. rotational symmetries in molecular structure); the label function mapping the input to some output (e.g. the image classification function is invariant to the location of the object in the image), the domain our data lives on (e.g. data living on a set is invariant to the permutation of items in the set), or even symmetries in the model's parameterization.

The key insight is that by encoding symmetry within our model architecture, we restrict the space of functions that can be represented to those that respect these symmetries. This makes models more performant, improves generalization, and can make learning more sample/data efficient.

Within genome assembly, we operate on input overlap graphs. By studying the symmetries of graphs by inspecting their invariances and equivariances, we are led to the @gnn machine learning architecture that is tailored to operate effectively on graph-structured data.
=== Permutation Invariance and Equivariance
Let $G = (V, E)$ be a graph such that $V$ is the set of nodes representing arbitrary entities. $E subset.eq V times V$ is the set of edges such that $(u, v) in E$ encodes relationships among these nodes/entities. The complete connectivity of $G$ has an algebraic representation $bold(A) in RR^(|V| times |V|)$, the adjacency matrix such that:
$ A_(u v) = cases(1", if"  space (u, v) in E,
                  0", if" space (u, v) in.not E) $

Now assume that each node $v$ is equipped with a node feature vector $bold(x)_v$. By stacking these per-node feature vectors, we get the node feature matrix $bold(X) = (bold(x_1), ..., bold(x_n))^T in |V| times k^"n"$, where $k^"n"$ is the node feature dimension. $bold(X) [v]$ corresponds to $bold(x)_v$.

Similarly, assume that each edge $(u, v) in E$ is equipped with an edge feature vector $bold(e)_(u v)$. The per-edge feature vectors are packed into an edge feature matrix $bold(E) in |V| times |V| times k^"e"$, where $k^"e"$ is the edge feature dimension. $bold(E) [u][v]$ corresponds to $bold(e)_(u v)$.

#let perm = $bold(P)$
#let permedge = $dash(bold(P))$
#let features = $bold(X)$
#let edgefeatures = $bold(E)$
#let adj = $bold(A)$

Given an arbitrary permutation matrix #perm, a function $f$ is said to be permutation _invariant_ iff
$f(perm features, permedge edgefeatures permedge^T, perm adj perm^T) = f(features, edgefeatures, adj)$, where $forall k in k^"e". thin permedge[u, v, k] = perm$. Likewise, a function $bold(F)$ is said to be permutation _equivariant_ iff $bold(F)(perm features, permedge edgefeatures permedge^T, perm adj perm^T) =  perm bold(F)(features, edgefeatures, adj)$. These correspond to the the automorphism group of a graph---the set of all symmetry operations preserving the graph's structure i.e. the vertices are permuted in such a way that the adjacency structure is preserved.

=== Graph Neural Networks <section:gnn>
#let neighborhood = $cal(N)$
Let $G = (V, E)$ be a graph, with $neighborhood_v = {u in V : (v,u) in E}$ representing the one-hop neighborhood  of node $v$, having node features $features_(neighborhood_v) = {{bold(x)_u: u in neighborhood_v}}$, and edge features $edgefeatures_(neighborhood_v) = {{bold(e)_(u v): u in neighborhood_v}}$, where ${{dot}}$ denotes a multiset.
We define $f$, the message passing function, as a local and permutation-invariant function over the neighborhood features $features_(neighborhood_v)$ and $edgefeatures_(neighborhood_v)$ as:
$ f(bold(x)_v, features_(neighborhood_v), edgefeatures_(neighborhood_v)) = phi.alt(bold(x)_v, plus.circle.big_(u in neighborhood_v) psi(bold(x)_u, bold(x)_v, bold(e)_(u v))) $ <eq:gnn_message_passing>

where $psi$ and $phi.alt$ are learnable message, and update functions, respectively, while $plus.circle$ is a permutation-invariant aggregation function (e.g., sum, mean, max). A permutation-equivariant @gnn layer $bold(F)$ is the local message passing function applied over all neighborhoods of $G$:
$ bold(F)(features, edgefeatures, adj) = mat(dash.em f(bold(x)_1, features_(neighborhood_1), edgefeatures_(neighborhood_1)) dash.em;
                               dash.em f(bold(x)_2, features_(neighborhood_2), edgefeatures_(neighborhood_2)) dash.em;
                               dots.v;
                               dash.em f(bold(x)_n, features_(neighborhood_n), edgefeatures_(neighborhood_n)) dash.em;) $

A @gnn consists of sequentially applied message passing layers.

=== Expressivity of Graph Neural Networks
Graphs $G_1$ and $G_2$ are considered isomorphic if they encode the same adjacency structure under some permutation of their nodes. Although @gnn:pl are powerful graph processing tools, they are unable to solve all tasks on a graph accurately.

A @gnn is able to distinguish two non-isomorphic graphs $G_1$ and $G_2$, if it maps them to differing graph embeddings (in $RR^d$, for some arbitrary dimension $d in NN$) i.e. $bold(h)_(G_1) eq.not bold(h)_(G_2)$. The ability to distinguish non-isomorphic graphs is important as without this capability, solving a task requiring discriminating between them is unachievable. The graph isomorphism problem is challenging, with no known polynomial-time algorithm.

The expressive power of a @gnn is assessed by the set of graphs that they can distinguish (mapping them to different embeddings if, and only if, the graphs are non-isomorphic). Formally, assume that the set of all @gnn:pl is given by the set $PP$, and the set of all graphs is given by the set $GG$. Now, further assume that $P_1, P_2 in GG$ are arbitrary @gnn:pl, and that the set of graphs distinguishable by $P_1$ and $P_2$ are $DD_(P_1), DD_(P_2) subset.eq GG$). We then define the expressive power partial ordering over $PP$, $prec.eq$ , as:
$
  P_1 prec.eq P_2 <==> DD_(PP_1) subset.eq DD_(PP_2)
$
and consequently, we also have $P_1 prec P_2 <==> DD_(PP_1) subset DD_(PP_2)$.

It has been proven @gin-paper that the @gnn formulation laid out in @section:gnn is at most as powerful at distinguishing non-topologically identical graphs as the 1-@wl test displayed in @alg:1-el (note the similarity to @gnn message passing in @eq:gnn_message_passing).

#let wl_test = [#set text(size: 0.9em)
#algorithm-figure("1-Weisfeiler-Lehman-Test", {
  import algorithmic: *
  Procedure([1-@wl], ([$G_1 = (V_1, E_1)$],[$G_2 = (V_2, E_2)$],), {
    Comment[Assign identical starting colors to each node in both graphs]
    Assign[$forall u in V_1, thin thin thin thin c_(u, G_1)$][$c_0$]
    Assign[$forall v in V_2, thin thin thin thin c_(v, G_2)$][$c_0$]
    State[]
    While([colors are not stable], {
      Comment[Update each node's color]
      Comment[Note that $"HASH"$ is some color hashing function]
      State[]
      Assign[$forall u in V_1, thin thin thin thin c_(u, G_1)^(t)$][$"HASH"(c_(u, G_1)^(t - 1), {{c_(w, G_1)^(t - 1)}}_(w in cal(N)_u))$ #v(1em)]
      Assign[$forall v in V_2, thin thin thin thin c_(v, G_2)^(t)$][$"HASH"(c_(v, G_2)^(t - 1), {{c_(w, G_2)^(t - 1)}}_(w in cal(N)_v))$]
    })
    State[]
    If([${{c_(u, G_1)^(t)}}_(u in V_1) eq.not {{c_(v, G_1)^(t)}}_(v in V_2)$], {
      [#v(0.5em) return _not_ isomorphic #v(0.2em)]
    })
    Else({
      [return _possibly_ isomorphic]
    })
  })
}) <alg:1-el>]
#place(top + center)[#wl_test]

@gnn expressivity is an important topic for solving problems on graphs that require identifying and differentiating graph structure. Since the layout problem in genome assembly is fundamentally about graph structure, this is a critical area of interest.

== Mamba Selective State Space Model
Mamba is derived from the class of @s4 models @mamba, combining aspects of recurrent, convolutional, and classical state space models. While @s4 models have a recurrent formulation, a parallelizable convolutional operation applied to a sequence yields the identical result, making them much faster than previous @rnn architectures, such as @lstm @lstm-review networks.

// #grid(
//   columns: (1.3fr, 1fr),
//   gutter: 2em,
//   align: center + horizon,
//   grid(
//     rows: 2,
//     gutter: 3em,
//     [#image("graphics/s4_cont.png")],
//     [#image("graphics/rnn.png", width: 6cm)]
//   ),
//   [#image("graphics/scan.png")],
// )

#place(top + center)[#subpar.grid(
    columns: (0.32fr, 0.95fr, 1fr),
    align: center,
    gutter: 2em,
    show-sub-caption: sub-caption-styling,
    figure(image("graphics/s4_continuous.svg", height: 4cm), caption: [S4 Cont.]), <fig:s4_continuous>,
    figure(image("graphics/s4_discrete_recurrent.svg", height: 4cm), caption: [S4 Discrete (Recurrent)]), <fig:s4_discrete_recurrent>,
    figure(image("graphics/s4_discrete_convolutional.svg", height: 4cm), caption: [S4 Discrete (Convolutional)]), <fig:s4_discrete_convolutional>,
    caption: [Illustration of the continuous-time @s4 model in (a). (b) and (c) show how the discrete @s4 model can be represented equivalently by recurrence and convolution.]
  )]

Moreover, these models have principled mechanisms for long-range dependency modelling @ssm-long-range, and perform well in benchmarks such as Long Range Arena @long-range-arena. Their speed and ability to capture long-range dependencies make them compelling for sequence modelling tasks.

Formally, S4 models, defined with continuous-time parameters $(Delta, bold(A), bold(B), bold(C))$ can be formulated as follows:
$ h'(t) = bold(A) h(t) + bold(B) x(t) #h(5em) y(t) = bold(C) h(t) $
// $ y(t) = bold(C) h(t) $

These equations refer to a continuous-time system, mapping a _continuous _ sequence $x(t) in bb(R) arrow.r y(t) in bb(R)$, through an implicit hidden latent space $h(t) in bb(R)^N$ (illustrated in @fig:s4_continuous). For discrete data however, like a sequence of bases in a read, these equations need to be discretized. Before detailing the discretization procedure, we note that having an underlying continuous-time system is beneficial as we inherit beneficial properties of continuous-time dynamics---key is smoother encoding of long-range dependencies and memory. Moreover, there are well-established connections between discretization of continuous time systems and @rnn gating mechanisms.


Discretization is performed using the step size parameter $Delta$, transforming the continuous-time parameters $(Delta, bold(A), bold(B))$ into discrete-time parameters $(bold(dash(A)), bold(dash(B)))$ through a discretization rule. $Delta$ can be viewed as a more generalized version of the gating mechanism found in @rnn:pl. Mamba Selective State Space model uses zero-order hold as its discretization rule, where $dash(A) = exp(Delta A)$ and $dash(B) = (Delta A)^(-1)(exp(Delta A) - I) dot Delta B$). This yields a new set of discrete equations:
$ h_t = dash(bold(A))h_(t - 1) + dash(bold(B)) x_t #h(5em) y_t = bold(C)h_t $
// $ y_t = bold(C)h_t $

Through repeated application of the recurrence relation, and simplification via the @lit property (which states that $(Delta, bold(A), bold(B))$ and consequently $(bold(dash(A)), bold(dash(B)))$ remain constant for all time-steps), the system can be equivalently expressed as a 1-dimensional convolution (see @fig:s4_discrete_recurrent and @fig:s4_discrete_convolutional for illustration) over the sequence $x$ with kernel $bold(dash(K))$ ($star$ denotes the convolution operation):
$ bold(dash(K)) = (C dash(B), C dash(A) dash(B), ..., C dash(A)^k dash(B), ...) #h(5em) y = x star bold(dash(K)) $
// $ y = x star bold(dash(K)) $

Since @s4 models have fixed parameters with respect to the inputs, they cannot perform content-based reasoning, which is essential for language, or genome modelling tasks. To address this, Mamba extends the S4 formulation by incorporating _selectivity_---the ability to select data in an input-dependent manner---helping filter out irrelevant data, and keep relevant information indefinitely, by making the parameters functions of the input. However, this breaks the time- and input-invariance (@lit) property allowing fast convolution-based calculation. To compensate, convolution is replaced with a scan/prefix sum operation (@fig:recurrent and @fig:parallel_scan show how the scan/prefix sum algorithm produces the same result as recurrence). 

Selecting data in an input-dependent manner, combined with the scan/prefix sum algorithm, results in the Mamba _Selective_ State Space Model (@fig:mamba_official).

#place(top + center)[#subpar.grid(
  columns: (1fr, 1fr),
  gutter: 0em,
  show-sub-caption: sub-caption-styling,
  figure(image("graphics/recurrent.svg", height: 4.5cm, fit: "contain"), caption: [Recurrent formulation for generating #linebreak() @s4 hidden states]), <fig:recurrent>,
  figure(image("graphics/parallel_scan.svg", height: 6.5cm, fit: "contain"), caption: [Scan/Prefix sum formulation]), <fig:parallel_scan>,
  caption: [Illustration of how the scan/prefix sum algorithm (b) produces the same result as the recurrent (sequential) formulation (a) in generating the @s4 hidden states in parallel. Note that, for example, $x_3$'s calculation begins before $x_2$ has been fully calculated in (b).]
)]
#v(-0.5em)
#figure(image("graphics/mamba_official.png", height: 3.5cm), caption: [Mamba's selection mechanism in #text(fill:blue.darken(50%))[blue] alters parameters in an input-dependent manner. Diagram from #cite(<mamba>, form: "prose").]) <fig:mamba_official>


// Even the reference human genome has 100s of assembly gaps that are 100s of Mb (megabases) of highly repetitive or recently duplicated sequences.

// Current (long-read) sequencing technologies produce contiguous reads ranging from 100b--10s of kb.
//   - Chromosomes are $>$ 10 Mb, and can be 1 Gb long.
//   - De Novo assembly requires individual reads that cover the genome multiple times, pieced together using overlaps between them.

// Key properties of reads (that are often traded-off) are length, accuracy, and evenness of representation:
//   - We focus on PacBio HiFi reads due to combination of their high accuracy (above 99.5%) and length (15000--25000 base pairs).

// OLC algorithm overview

// We are focussing on the Layout phase, so provide more details on:
// + With perfect reads, the genome would just be the Hamiltonian path through this graph
// + But we have errors due to [state reasons]
// + Traditionally heuristics used to clean-up this graph---transitive edges, dead-ends, bubble---perhaps some details on such heuristics
// + In complex genomic regions, these don't work---give examples of these---contemporary assemblers would cut-out these regions, leading to fragmented assemblies
// + In complex regions, we currently rely on manual effort, which obviously does not scale

// GNN background:
// + Permutation invariance and equivariance
// + General form
// + GCN, GAT
// + Expressiveness on GNNs---also link to why expressiveness/detection of patterns is important to this problem
// + Problems with previously used Graph Normalization techniques---reduce expressiveness of the neural network

// Mamba SSM background and linear-time sequence modelling

#pagebreak()

// = Related work

// #pagebreak()

= Design and implementation
In this chapter, we detail our training and inference setup explaining: the generation of overlap graphs; ground-truth label generation; data augmentation and partitioning; the training objective, and genome reconstruction via greedy decoding. This is followed by a discussion of  the various model architectures tested including: more advanced @gnn layers than those used in prior work @lovro @lovrolong; our integration method of raw nucleotide-level read data into the model, and a graph-adaptive normalization technique.

// Ultra-long reads have have demonstrated significant advantages in resolving complex artifacts in overlap graphs and repeating regions in genomes, and prior work @lovro @lovrolong has presented @gnn:pl as a viable method for improving the layout phase in the @olc algorithm. We are interested in investigating the utility of @gnn:pl in leveraging ultra-long read data to advance the capabilities of neural genome assembly methods.

// The integration of ultra-long reads with conventional long-read data alters the structural properties of the resulting overlap graphs. This motivates exploring alternative @gnn architectures that may better exploit the additional information available. In this chapter, we detail our training and inference setup, discuss the various @gnn architectures tested, and explain our integration method of raw read data into the model.
#place(top + center)[#figure(
  image("graphics/overview.svg"),
  caption: [(A) Simulated @pacbio @hifi reads are generated from a reference genome via `PBSIM3`. Additional @ont @ul reads may also be generated. Alternatively, real read data can also be provided. The reads are then passed to `Hifiasm`, which constructs the corresponding overlap graph. (B) Ground-truth edge labels are computed corresponding to the optimal assembly. During training only, the overlap graph is masked and partitioned. Masking allows for data augmentation by simulating varying read coverage (number of times each base is sampled on average) from $times #h(0em) 30$ to $times #h(0em) 60$. Partitioning is required to fit onto @gpu memory. (C) Features are extracted from the overlap graph according to the model used, and edge probability predictions are made by the model. Note the reversed compliment is only used by some of the @gnn models. The loss in computed relative to the ground-truth labels. (D) The genome is reconstructed via greedy decoding.]
) <fig:overview>]

== Training and inference setup
We detail the training and inference pipeline next. A detailed illustration can be found in @fig:overview.

=== Generating the overlap graph
The first step of generating an overlap graph is gathering the raw read data. Since we are unable to produce our own sequencing data, reads from the `CHM13v2` @t2t human genome assembly (`BioProject PRJNA559484` @chm13-acrocentric) are instead simulated. This simulation is performed using a utility called `PBSIM3` @pbsim3 that emulates the read profile of @pacbio @hifi long-reads according to `fastq` data (`fastq` is a format for storing the sequencing data, in addition to per-base quality scores that are crucial for our simulation) from the sequencing of the `HG002` draft human reference @hg002-github. When simulating reads, a $times #h(0em) 60$ coverage factor is used (enough reads to cover the genome $60$ times over). @sec:datasets details the sections of the genome used for training, validation, and testing in more detail.

// For training, we choose chromosomes 19 and 15, representing both non-acrocentric, and acrocentric chromosomes. An acrocentric chromosome is one where the centromere, the region of a chromosome that holds sister chromatids together, is not located centrally on the chromosome, but towards one end. For validation and test, we likewise choose chromosomes 11 and 22, and chromosomes 9 and 21, respectively. Note that the chromosomes chosen for both training and evaluation, represent the most difficult ones during assembly due to the tangles often present in their real-life overlap graphs. Additionally, the centromeric region of each of these chromosomes is extracted for generating reads, where most assembly complexity arises @chm13-acrocentric. By training on only a small portion of the chromosomes present in the genome, we demonstrate the positive generalization capabilities of our neural method.

Once the reads are generated, `Hifiasm` @hifiasm-paper, a de novo assembler specifically designed for @pacbio @hifi read data, is used to generate the overlap graph. Note that no traditional graph simplification algorithms like transitive edge removal, dead-end trimming, or bubble removal, are applied. Also, it is important to note that the overlap graph produced is a symmetric overlap graph as the reads can belong to either strand of the @dna. The symmetric overlap graph consists of one graph, and its dual that contains a duplicate set of nodes representing the same reads, but with the edges reversed. This is due to an interesting property during sequencing where reads from the dual @dna strand are sequenced in reverse order along the length of the @dna.

=== Overlap graph ground-truth label generation
We refer to whether an edge belongs to the final assembly as a boolean _label_, which is the target the @gnn aims to predict. There are two conditions for an edge to be valid (i.e. labeled true): (1) The reads the edge states overlap must be sampled from the same strand of @dna (@eq:same_strand) and have a valid overlap (@eq:valid_overlap), and (2) the edge must not lead to a read that is a dead-end. Formally, the first condition states that for reads $A$ and $B$, with edge $A -> B$:
$
A_"strand" = B_"strand" "(same strand)"
$ <eq:same_strand>
#set math.cases(reverse: true)
$
cases(A_"start" &< B_"start",
A_"end" &> B_"start",
A_"end" &< B_"end") "(valid overlap)"
$ <eq:valid_overlap>

where $X_"strand"$, $X_"start"$, $X_"end"$ refer to the strand, starting, and ending positions in the actual genome for some read $X$. Edges not satisfying this first condition are marked with the label false. Note that since the reads are simulated, we know the true strand and positions along the genome they are sampled from. To find the edges also satisfying the second property, we follow the algorithm laid out in @alg:find-optimal-edges (and used by prior neural genome assembly work @lovro).
#let algorithm_1 = [#set text(size: 0.9em)
#algorithm-figure("Find-Optimal-Assembly", {
  import algorithmic: *
  Procedure("Find-Optimal-Edges", ([_overlap-graph_],), {
    Comment[Initialize the set of edges belonging to the optimal assembly]
    Assign[_optimal-edges_][${}$]
    State[]
    For([_connected-component_ *in* _overlap-graph_], {
      Comment[Decompose the connected-component into nodes and edges]
      Assign[$V$, $E$][_connected_component_]
      State[]
      Comment[Start search from the (node corresponding to) read #linebreak() at the lowest position along the genome sequence]
      Assign[_lowest-read-node_][$"argmin"_(v thin in thin V)$ #FnInline[get-read-start-loc-for-node][$v$]]
      State[]
      Comment[Perform the forward @bfs]
      Assign[_visited-nodes-forward_][${}$]
      Assign[_visited-edges-forward_][${}$]
      Assign[_visited-nodes-forward_, _visited-edges-forward_][#linebreak() #h(2em) #FnInline[@bfs][_connected-component_, start=_lowest-read-node_]]
      State[]
      Comment[Start the reverse search from the (node corresponding to) _visited_ read #linebreak() at the highest position along the genome sequence]
      Assign[_highest-read-node_][$"argmax"_(v thin in italic("visited-nodes-forward"))$ #FnInline[get-read-start-loc-for-node][$v$]]
      State[]
      Comment[Perform the reverse @bfs]
      Assign[_visited-nodes-backward_][${}$]
      Assign[_visited-edges-backward_][${}$]
      Assign[_visited-nodes-backward_, _visited-edges-backward_][#linebreak() #h(2em) #FnInline[@bfs][_connected-component_, start=_highest-read-node_]]
      State[]
      Comment[The edges belonging to the final assembly are traversed by both @bfs:pl]
      Assign[_optimal-edges_][_optimal-edges_ $union$ (_visited-edges-forward_ $inter$ _visited-edges-backward_)]
    })
    State[]
    Return[_optimal-edges_]
  })
}) <alg:find-optimal-edges>]
#place(top + center)[#algorithm_1]

We start from the edge whose starting read is at the lowest position along the genome and perform a @bfs from it, storing the visited nodes. From this set of visited nodes, another @bfs is performed starting from the node representing the read at the highest genomic position. Edges traversed by both of the @bfs:pl belong to the optimal assembly (called _optimal-edges_). If there are multiple connected components in the overlap graph, the process is repeated. The _optimal-edges_ are labeled as belonging to the final assembly (true)---all other edges are labeled false.

=== Overlap graph masking and partitioning
Masking and partitioning are performed during training only, with the entire graph used for performing inference. Masking is performed as a form of data augmentation to cheaply produce different sets of reads and the corresponding overlap graph. For every training step, $0$--$20%$ of the overlap graph's nodes, and corresponding edges, are removed. This simulates varying levels of read coverage (number of times each base in the genome is sampled on average) up to the original $times #h(0em) 60$.

Additionally, since the entire overlap graph contains $>100,000$ nodes, and cannot fit onto @gpu memory, `METIS` @metis-paper partitioning is used to divide the overlap graph. Note that inference is performed on the @cpu, which is able to access the system's abundant main memory, and so graph partitioning is not required.

=== Feature extraction and running the models
We leave discussion of the node and edge features extracted from the overlap graph and raw read data to later sections, since they depend on the model architecture used. The models then take the overlap graph, and these node and edge features as input, producing for each edge, a probability of that edge belonging to the final assembly.

=== Loss function
Assume that we are given a _symmetric_ overlap graph $G = (V, E)$. For some nodes $A, B in V$, if the edge $e: A -> B$ leads to the optimal solution, then its virtual sister edge  $e': B' -> A'$ also leads to the optimal solution on the reverse strand of the @dna ($A', B' in V$ complete the virtual pairs of $A$ and $B$ respectively) @lovrolong. This is a consequence of the dual-stranded nature of @dna.

Recall that we are using a @gnn to predict the probability of an edge $e$ belonging to the final assembly. We encode the symmetry property described above for the paired edge $e'$ into the loss:
$
  cal(L)(e, e') = "BCE"_"logits" (l_e, y_e) + "BCE"_"logits" (l_e', y_e') + underbrace(alpha abs(l_e - l_e'), "Symmetry Loss")
$
where $cal(L)$ is the complete loss function used, $"BCE"_"logits"$ is Binary Cross Entropy with Logits loss, $l_e, l_e' in RR$ are the logits for edges $e, e'$ predicted by the model respectively, $alpha$ is a parameter controlling the contribution of the symmetry loss to the total loss $cal(L)$, and $abs(...)$ corresponds to taking the absolute value. @fig:overview (C) illustrates symmetry loss.

#let modelexplanation = it => [
  #box(fill: blue.lighten(90%), inset: 1em, stroke: blue, radius: 1em, width: 100%)[#it]
]

#modelexplanation[
  We augment the standard Binary Cross Entropy loss with Symmetry loss to penalize the model assigning different probabilities to edges belonging to the same virtual pair, due to the dual-stranded nature of @dna.
]

=== Reconstructing the genome via greedy decoding
After assigning a probability to each edge representing its likelihood of belonging to the final assembly, we apply a greedy decoding algorithm (detailed below) to extract contigs---sets of overlapping @dna fragments that together reconstruct a contiguous portion of the genome:

#let algorithm_2 = [#set text(size: 1em)
#algorithm-figure("Genome reconstruction via greedy decoding", {
  import algorithmic: *
  Procedure("Greedy-Decode-Contigs", ([_overlap-graph_],[_edge-probabilities_]), {
    Assign[_final-assembly_][${}$]
    State[]
    While([_overlap-graph_ contains unvisited nodes], {
      Comment[Sample $B$ starting edges using an empirical distribution given by _edge-probabilities_]
      Assign[$E$][${e_1, ..., e_B}$, where $e_i$ is an edge with probability $bb(P)(e_i) = italic("edge-probabilities")[$e_i$]$]
      State[]
      For([$e_i in E$], {
        Comment[Initialize path greedily decoded from this edge]
        Comment[Although the path $p_i$ is a list of edges, allow checking if a node is in the path for notational ease]
        Assign[$p_i$][[$e_i$]]
        State[]
        Comment[Greedy forward search from $v_i$ (target node of $e_i: u_i -> v_i$)]
        Comment[New edge(s) to be traversed must be unvisited, and lead to an unvisited node]
        While([unvisited outgoing edge from last node in $p_i$, $v_k$, exists], {
          Comment[Choose outgoing edge from $v_k$ with highest probability]
          Assign[$e_k$][$"argmax"_("outgoing edge" e_k "from" v_k) bb(P)(e_k) $]
          Comment[Append this edge to extend the path $p_i$]
          Assign[$p_i$][$p_i$ + [$e_k$]]
        })
        State[]
        Comment[Greedy backward search from $u_i$ (source node of $e_i: u_i -> v_i$)]
        Comment[New edge(s) to be traversed must be unvisited, with an unvisited source node]
        While([unvisited incoming edge to first node in $p_i$, $v_j$, exists], {
          Comment[Choose incoming edge from $v_j$ with highest probability]
          Assign[$e_j$][$"argmax"_("incoming edge" e_j "from" v_j) bb(P)(e_j) $]
          Comment[Prepend this edge to extend the path $p_i$]
          Assign[$p_i$][[$e_j$] + $p_i$]
        })
        // State[]
        // Comment[Mark transitive nodes as visited]
        // For([node $v in.not p_i$], {
        //   If(cond: [
        //     #FnInline[predecessor][$v$] $in p_i and$ #FnInline[successor][$v$] $in p_i $ #linebreak() $and e:$ #FnInline[predecessor][$v$] $->$ #FnInline[successor][$v$] $in p_i$
        //   ], {
        //     FnInline[mark-node-visited][$v$]
        //   })
        // })
      })
      State[]
      Comment[Keep the longest path]
      Assign[_longest-path_][$"argmax"_(p_i)$ #FnInline[length][$p_i$]]
      State[]
      Comment[Convert the set of reads in the _longest-path_ into a contig]
      Assign[_contig_][#FnInline[to-contig][_longest-path_]]
      State[]
      Comment[Add the _contig_ to the _final-assembly_]
      Assign[_final-assembly_][_final-assembly_ $union$ _contig_]
      State[]
      Comment[Nodes (and edges) forming the contig cannot be reused to avoid duplicating regions]
      State[#FnInline[mark-nodes-visited][_longest-path_]]
      State[]
      Comment[Stop when the length of the longest contig found falls below a fixed threshold]
      If([#FnInline[length][_longest_path_] $<$ _min-contig-length_], {
        State[break]
      })
    })
    State[]
    Return[_final-assembly_]
  })
}) <alg:greedy-decode-contigs>]
// #place(top + center)[#algorithm_2]

Recall that we are interested in finding a Hamiltonian path through the overlap graph to recover the genome. In an ideal scenario, where all neural network edge predictions are accurate and the graph contains no artifacts, a simple greedy traversal (forwards and backwards) starting from any positively predicted edge would suffice to reconstruct the genome. However, due to prediction errors and noise in the graph, neither of these conditions are met in practice, and so we use the greedy decoding algorithm shown in @alg:greedy-decode-contigs (@app:algorithms), and illustrated in @fig:overview(D) (and used by prior neural genome assembly work @lovro).

@alg:greedy-decode-contigs first samples multiple high-probability seed edges and then greedily chooses a sequence of edges both forwards and backwards from each seed edge, forming a path through the assembly graph. The longest resulting path is selected and overlapping reads along that path merged into a contig. Nodes along the selected path are marked as visited to prevent their reuse in subsequent searches, and the process repeats until no path above a fixed length threshold can be found.

== Model architecture
We describe the feature encoding, processing @gnn layers, and decoding, together forming the complete model in this section.
=== Standard input features <sec:standard_input_features>
Assume we are given an overlap graph $G = (V, E)$. For two overlapping reads $r_i$ and $r_j$, represented by nodes $v_i$ and $v_j$, and connected by edge $e_(i j): i -> j$ the edge feature $z_(i j) in bb(R)^2$ is defined as follows:
$ z_(i j) = &("normalized" italic("overlap length") "between" r_i "and" r_j, \
&"normalized" italic("overlap similarity") "between" r_i "and" r_j) $

$ italic("overlap similarity") = (italic("overlap length") - italic("edit distance")) / italic("overlap length") $

where normalized refers to standard z-scoring (zero mean and unit standard deviation) over the set of all edges $E$ in the overlap graph. Note that the _edit distance_ is given by approximate string matching between the overlapping suffix of $r_i$ and prefix of $r_j$.

The standard node input edge features $x_i in bb(R)^2$ are given as:
$ x_i = (italic("in-degree") "of" v_i, italic("out-degree") "of" v_i) $

noting that these node features are calculated before graph masking and partitioning during training.

Henceforth, $z_(i j)$ and $x_i$ will together be referred to as the standard input features.

=== Standard input embedding <sec:standard_input_embedding>
The embedding of the standard input features into the initial hidden representations $h_i^0 in bb(R)^D$ for node $i$ at layer $0$, and $e_(i j)^0 in bb(R)^D$ for the edge $i -> j$ (where $i$ and $j$ are nodes) at layer 0 are computed as:
$
  h_i^0 &= W_2^"n" (thin "ReLU" (W_1^"n" x_i + b_1^"n")) + b_2^"n" \
  e_(i j)^0 &= W_2^"e" (thin "ReLU" (W_1^"e" z_(i j) + b_1^"e")) + b_2^"e"
$
where all $W^"n"$ and $b^"n"$, and $W^"e"$ and $b^"e"$ represent learnable parameters for transforming the node and edge features respectively ($W_1^"n", W_1^"e" in bb(R)^(D times 2)$, $W_2^"n", W_2^"e" in bb(R)^(D times D)$, and $b_1^"n", b_1^"e", b_2^"n", b_2^"e" in bb(R)^D$), and $D$ is the hidden dimension.

We refer to this formulation for $h_i^0$ and $e_(i j)^0$ as the standard input embedding.

=== SymGatedGCN
The @symgatedgcn formulation is taken from prior neural genome assembly work @lovro, and acts as a baseline model for this project. Let the hidden representations of node $i$ and edge $e_(i j): i -> j$ at layer $l in {1, 2, ..., L}$ be $h_i^l$ and $e_(i j)^l$ respectively. Additionally, let $p$ denote node $i$'s predecessors and $s$ denote its successors. Each @symgatedgcn layer then transforms the hidden node and edge embeddings as follows:
#let relu = [$"ReLU"$]
#let norm = [$"Norm"$]
$
  h_i^(l + 1) = h_i^l + #relu (#norm (A_1^l h_i^l + sum_(p -> i) eta_(p i)^("f", l + 1) dot.circle A_2^l h_p^l + sum_(i -> s) eta_(i s)^("b", l + 1) dot.circle A_3^l h_s^l))
$ <eqn:symgatedgcn-update-hidden>
$
  e_(i j)^(l + 1) = e_(i j)^l + #relu (#norm (B_1^l e_(i j)^l + B_2^l h_i^l + B_3^l h_j^l))
$ <eq:edge_features>
where all $A, B in RR^(D times D)$ are learnable parameters with hidden dimension $D$, $dot.circle$ denotes the Hadamard product, #relu stands for Rectified Linear Unit, and #norm refers to the normalization layer used---this is either InstanceNorm @instancenorm, or @granola (@sec:granola). Note that the standard input embeddings (@sec:standard_input_embedding) are used for $h_i^0$ and $e_(i j)^0$. $eta_(p i)^("f", l)$ and $eta_(i s)^("b", l)$ refer to the forward, and backward gating functions respectively. The edge gates are defined according to the GatedGCN:
$
  eta_(p i)^("f", l) = sigma(e_(p i)^l) / (sum_(p' -> i) sigma (e_(p' i)^l) + epsilon.alt) in [0, 1]^D, #h(2.5em) eta_(i s)^("b", l) = sigma(e_(i s)^l) / (sum_(i -> s') sigma (e_(i s')^l) + epsilon.alt) in [0, 1]^D
$
where $sigma$ represents the sigmoid function, $epsilon.alt$ is a small value added to prevent division by 0, and $j' -> i$ represents all edges where the destination node is $i$. Likewise, $i -> k'$ represents all edges where the source node is $i$.

#modelexplanation[
  Most conventional @gnn layers are designed to operate on undirected graphs, and therefore do not account for directional information intrinsic to overlap graphs. This limitation is problematic, since the overlap graph encodes the directional path reflecting the linear structure of the genome from start to end. @symgatedgcn aims to address this lack of expressivity by distinguishing the messages passed along the edges $(sum_(p -> i) eta_(p i)^("f", l + 1) dot.circle A_2^l h_p^l)$, to those passed along the reversed direction of the edges $(sum_(i -> s) eta_(i s)^("b", l + 1) dot.circle A_3^l h_s^l)$.
]

=== GAT+Edge <sec:gat-edge>
The standard @gat @gat-paper architecture only focusses on node features, and so we extend this architecture to update edge features, include them in the attention calculation, and use them to also update the node features.

First, updated edge features are calculated identically to @symgatedgcn (@eq:edge_features).

In contrast to the @gat architecture with a single shared attention mechanism, there are now two mechanisms, $a^"n"$ and $a^"e"$, which compute the attention coefficients for nodes and edges respectively ($a^"n", a^"e": RR^D times RR^D times RR^D -> RR$). Each mechanism is implemented via separate, single-layer feed-forward neural networks. The attention coefficients are given as follows:
$
  c_(i j)^"n" &= a^"n" (h_j^l || e_(j i)^l || h_i^l) #h(5em)
  c_(i j)^"e" &= a^"e" (h_j^l || e_(j i)^l || h_i^l) \
$
where $c_(i j)^"n"$ indicates the importance of node $j$'s features to node $i$, and $c_(i j)^"e"$ indicates the importance of the edge $e_(j i): j -> i$ to node $i$. $||$ denotes the concatenation operator along the hidden dimension. These coefficients are then normalized over all $j$ to make them comparable across nodes, via softmax:
$
  alpha_(i j)^"n" &= "softmax"_j (c_(i j)^"n") = (exp (c_(i j)^"n")) / (sum_(k in neighborhood_i) exp (c_(i k)^"n")) \
  alpha_(i j)^"e" &= "softmax"_j (c_(i j)^"e") = (exp (c_(i j)^"e")) / (sum_(k in neighborhood_i) exp (c_(i k)^"e")) \
$
The updated node features are then calculated by first weighing the node and edge features by their corresponding normalized attention coefficients. Next, these node and edge features are concatenated, and passed through another single-layer feed-forward neural network, #smallcaps[Mix-Node-Edge-Info]:
$
  
  h_i^(l + 1) = #smallcaps[Mix-Node-Edge-Info] (lr(sigma (sum_(j in neighborhood_i) alpha_(i j)^"n" bold(W)^"n" h_j^l ) ||) thin sigma (sum_(j in neighborhood_i) alpha_(i j)^"e" bold(W)^"e" e_(j i)^l ))
$
where $bold(W)^"n", bold(W)^"e" in RR^(D times D)$ are parameterized weight matrices. || denotes concatenation.

#modelexplanation[
  We refer to our custom attention-based formulation, which incorporates edge features, as @gatedge. Although there exist alternative @gat implementations incorporating edge features into the attention calculation, like PyTorch Geometric's GATConv @pytorch-geometric, GATConv does not allow edge to node message passing.
  
  Furthermore, a key theoretical limitation of the @gcn and @symgatedgcn architectures is that the transformations applied to the different nodes and edges in the neighborhood are the same. @gatedge, like the original @gat architecture it extends, implicitly enables assignment of different importances to nodes (and with @gatedge, edges) of the same neighborhood @gat-paper. This could improve the model's adaptability to various overlap graph artifacts. Additionally, @gatedge remains a computationally efficient architecture.
]

=== SymGAT+Edge <sec:symgat-edge>
With the design of this architecture, we aim to combine the symmetry mechanism from @symgatedgcn with the @gatedge architecture mentioned previously. This is done by first calculating the updated edge features $e_(i j)^(l + 1)$ identically to @symgatedgcn (@eq:edge_features).

Next, a copy of the input graph $G = (V, E)$ is made, $G_"rev" = (V, E_"rev")$, such that:
$ forall i, j in V. thick i -> j in E <==> j -> i in E_"rev" $
$G_"rev"$ is equivalent to the original graph $G$, with the direction of all edges reversed. @gatedge then individually takes $G$ and $G_"rev"$ as input, producing a pair of new node features $h_i^("f", l + 1)$ and $h_i^("b", l + 1)$ respectively. These are then combined to produced to new hidden node state as follows:
$
  h_i^(l + 1) = h_i^l + #relu (#norm (h_i^("f", l + 1) + h_i^("b", l + 1)))
$

#modelexplanation[
  Integrating the symmetry mechanism from @symgatedgcn into @gatedge, to form @symgatedge, helps to increase expressivity as messages passed along edges cannot be distinguished from messages passed along the reversed direction by the attention mechanism either. The model is just provided with a graph, with no information regarding what constitutes a forward and reverse edge.
]

=== SymGatedGCN+Mamba <sec:symgatedgcn-mamba>
The standard input features (@sec:standard_input_features) used in prior work on neural genome assembly extract normalized overlap length and similarity from pairs of overlapping reads. However, the models have access to only these summary statistics, not the raw nucleotide read data, which could enable the model to extract more complex features, for example by capturing some notion of what is biologically plausible.

While standard encoder-only Transformers @transformer-paper are the contemporary choice for sequence-to-embedding tasks like this @bert-paper, a fundamental drawback makes them unsuitable---their quadratic complexity with respect to the sequence length. Subquadratic-time attention mechanisms have been unable to match the performance of the original attention mechanism on modalities such as language @mamba. Each read is upto $10s$ of @kb long for @pacbio @hifi reads, and there are $1000$s of reads even in the partitioned overlap graph used during training (note that we cannot partition the graph to an arbitrarily small number of nodes without sustaining major losses in performance as context around the graph artifact is lost).

On the other hand, @rnn architectures such as @lstm @lstm-review have linear complexity, but have traditionally struggled with modelling such long sequences. The key to the efficacy of Transformers, is the self-attention mechanism's ability to effectively route information from across the sequence, regardless of the distance.

As a result, we turn to the Mamba architecture, which with its selectivity mechanism and parallel scan implementation, is able to model complex, long sequences, without the computational cost of Transformers.

Additionally, another issue mitigated by the use of Mamba is that there is no canonical tokenization for a sequence of nucleotides. Operating directly on the nucleotide sequence is important for de novo sequencing, where we have no knowledge of the underlying genome, due to the absence of a reference. The Mamba model has been previously shown to operate well directly on nucleotide sequences on tasks involving @dna modelling @mamba.

The @symgatedgcn-mamba model uses the standard input features (from @sec:standard_input_features) in addition to the Mamba encoding of the reads as additional node features. Assume we are given an overlap graph $G = (V, E)$. For read $r_i in {"A, T, C, G"}^T$, represented by node $v_i$, the Mamba read encoding node feature $m_i in bb(R)^D$ is generated as follows ($D$ is size of the hidden dimension).

First, read $r_i in {"A, T, C, G"}^T$, which is a string of nucleotides of length $T$, is one-hot encoded to produce $r_i^"one-hot" in {0, 1}^(4 times T)$:
$
  r_(i, t)^"one-hot" = cases(
    (0, 0, 0, 1)^top "if " r_(i t) = "A",
    (0, 0, 1, 0)^top "if " r_(i t) = "T",
    (0, 1, 0, 0)^top "if " r_(i t) = "C",
    (1, 0, 0, 0)^top "if " r_(i t) = "G",
  )
$
where $t in {1, 2, ..., T}$ refers to the $t$th nucleotide in $r_i$. $top$ is the transpose operator. Next, the one-hot encoded representation is expanded to the hidden dimension $D$ via a learned parameter matrix $bold(W)^"expand" in RR^(D times 4)$, and then the read is encoded into $r_i^"encoded" in RR^(D times T)$ by #smallcaps[Mamba]:
$
  r_i^"encoded" = #smallcaps[Mamba] (bold(W)^"expand" r_i)
$
Note that $r_i^"encoded"$ is a matrix that varies in size with the length of the read. In order to obtain a fixed length, _whole_ read encoding, we take the last row of this matrix (indexing from 1):
$
  m_i = r_i^"encoded" [n]
$

The node embeddings are then updated to incorporate the information from the Mamba embedding, forming an intermediate node embedding $h_i '$:
$
  h_i ' = W_2 (thin #relu (W_1 (h_i^l || m_i) + b_1)) + b_2
$
where all $W, b$ represent learnable parameters ($W_1 in RR^(D times 2D)$, $W_2 in RR^(D times D)$, and $b_1, b_2 in RR^D$), and $D$ is the hidden dimension. $||$ denotes the concatenation operator.

The intermediate node embeddings, and the unmodified edge embeddings $e_(i j)^l$ are then passed to a @symgatedgcn layer, that outputs $h_i^(l + 1)$ and $e_(i j)^(l + 1)$, and acts as the output of @symgatedgcn-mamba.

#modelexplanation[
  The primary goal of @symgatedgcn-mamba is to explore whether the model can exploit the raw read data to generate new (node) features that are useful in resolving overlap graph artifacts. Mamba was chosen as the read encoding model of choice due to its near-linear time complexity, long-range dependency modelling capabilities, and promising results on adjacent @dna modelling tasks.
]

=== SymGatedGCN+MambaEdge <sec:symgatedgcn-mamba-edge>
We use the same Mamba read encoding node feature $m_i in bb(R)^D$ as in @symgatedgcn-mamba, but remove the dependency on standard edge features (@sec:standard_input_features). We no longer form an intermediate node embedding, but instead calculate an intermediate edge embedding $e_(i j) '$:
$
  e_(i j) ' = W_2 (thin #relu (W_1 (m_i || m_j) + b_1)) + b_2
$
where all $W, b$ represent learnable parameters ($W_1 in RR^(D times 2D)$, $W_2 in RR^(D times D)$, and $b_1, b_2 in RR^D$), and $D$ is the hidden dimension. $||$ denotes the concatenation operator.

The unmodified node embeddings $h_i^l$, and the intermediate edge embeddings are then passed to a @symgatedgcn layer, that outputs $h_i^(l + 1)$ and $e_(i j)^(l + 1)$, and acts as the output of @symgatedgcn-mambaedge.

// $
//   h_i^0 &= W_2^"n" (thin "ReLU" (W_1^"n" (x_i) + b_1^"n")) + b_2^"n" \
//   e_(i j)^0 &= W_2^"e" (thin "ReLU" (W_1^"e" (m_i || m_j) + b_1^"e")) + b_2^"e"
// $
// where all $W^"n"$ and $b^"n"$, and $W^"e"$ and $b^"e"$ represent learnable parameters for transforming the node and edge features respectively ($W_1^"n" in bb(R)^(D times 2), W_1^"e" in bb(R)^(D times 2D)$, $W_2^"n", W_2^"e" in bb(R)^(D times D)$, and $b_1^"n", b_1^"e", b_2^"n", b_2^"e" in bb(R)^D$), and $D$ is the hidden dimension. $||$ denotes the concatenation operator.

#modelexplanation[
  @symgatedgcn-mambaedge tests whether the model can recover the overlap length and similarity metrics used earlier, from raw read data (or alternatively generate even richer embeddings).
]

=== SymGatedGCN+RandomEdge <sec:symgatedgcn-randomedge>
We use the same formulation as @symgatedgcn-mambaedge, but replace the Mamba read encoding node feature $m_i in RR^D$ with a $D$-dimensional standard Normal distribution sample, such that $m_i ~ cal(N)(mu, Sigma)$, where $mu = bold(0) in RR^D$ and $Sigma = "diag"(bold(1)) in RR^(D times D)$.

#modelexplanation[
  @symgatedgcn-randomedge tests whether the features encoded by Mamba are useful, by offering a performance lower-bound point of comparison. At its simplest, an untrained Mamba model acts as a neural hashing function, and so @symgatedgcn-mambaedge should perform at least as well as @symgatedgcn-randomedge.
]

=== Graph Adaptive Normalization Layer <sec:granola>
Normalization has been shown to be critical for enhancing the training stability, convergence behavior, and overall performance of neural networks. Conventional normalization techniques such as BatchNorm @batch-norm, InstanceNorm @instancenorm, and LayerNorm @layernorm, have been widely adopted, but are not specifically designed to support graph-structured data. In fact, direct application of these standard normalization techniques can impair the expressive power of @gnn:pl, degrading performance significantly @granola-paper.

Increasing the depth of a @gnn by stacking additional layers theoretically expands the class of functions it can represent, but repeated message passing operations can lead to node embeddings becoming indistinguishable---an effect known as over-smoothing @pairnorm. This observation is also theoretically motivated---graph convolution can be viewed as a type of Laplacian smoothing @laplacian-smoothing, and so its repeated application in @gnn layers leads to embeddings having similar values. Mitigating over-smoothing is the primary motivator for many graph-specific normalization layers (e.g. PairNorm @pairnorm, DiffGroupNorm @diffgroupnorm).

Unfortunately, despite numerous efforts to develop graph-based normalization schemes, no method consistently outperforms the alternatives in all tasks and benchmarks. Furthermore, normalization schemes extended from traditional schemes such as BatchNorm and InstanceNorm, often reduce the expressive power of the @gnn @granola-paper.

The @granola @granola-paper authors postulate that there are two main reasons why these alternative schemes do not provide an unambiguous performance improvement across domains. Firstly, many methods use shared affine normalization parameters across all graphs, failing to adapt to input graph-specific characteristics. Secondly, special regard should be given to the expressive power of the normalization layer itself to distinguish non-isomorphic graphs, and then correctly tailor affine parameters to suit the input.

We begin by providing more details regarding the first point. Let $G = (bold(A), bold(X))$ denote a graph with $N in NN$ nodes, with adjacency matrix $bold(A) in RR^(N times N)$, and node feature matrix $bold(X) in RR^(N times D)$, where $D$ is the hidden embedding dimension. The pre-normalized node features for the $b$-th graph in a batch (batch size $B$), after the application of the $(l-1)$th @gnn layer is given by $tilde(bold(H))_b^l$:
$
  tilde(bold(H))_b^l = "GNN"_"Layer"^(l - 1) (bold(A)_b, bold(H)_b^(l - 1))
$
A general update rule to produce normalized node features $bold(H)^l$ over the batch of graphs is then given by: 
$
  bold(H)^l = phi.alt ("Norm" (tilde(bold(H))^l; l))
$
where $"Norm"$ refers to a normalization layer, and $phi.alt$ is some activation function. Normalization layers based on standardization of inputs are given by first shifting each input element $tilde(h)_(b, n, c)^l$ by mean $mu_(b, n, c)$, and scaling by standard deviation $sigma_(b, n, c)$. Then, the result is modified by some learnable affine parameters $gamma_c^l, beta_c^l in RR$:
$
  "Norm"(tilde(h)_(b, n, d)^l; tilde(bold(H))^l, l) = gamma_d^l (tilde(h)_(b, n, d)^l - mu_(b, n, d)) / sigma_(b, n, d) + beta_d^l
$ <eq:normalization_framework>

Note how the learnable affine parameters $gamma_d^l, beta_d^l$ do not depend on $b$, nor $n$. This means that they are not adaptive to the input-graph.

In BatchNorm, statistics are computed across all nodes and graphs in the batch, but separately across the hidden dimension. BatchNorm can then be derived by substituting the following mean and standard deviation into @eq:normalization_framework:
$
  mu_(b, n, d) = 1/(B N) sum_(b = 1)^B sum_(n = 1)^N tilde(h)_(b, n, d)^l #h(3em) sigma^2_(b, n, d) = 1/(B N) sum_(b = 1)^B sum_(n = 1)^N (tilde(h)_(b, n, d)^l - mu_(b, n, d))^2
$

#let granola = [#set text(size: 0.9em)
#algorithm-figure([@granola Layer], {
  import algorithmic: *
  Procedure([@granola], ([Node features $tilde(bold(H))_b^l in RR^(N times D)$ from $"GNN"_"Layer"^(l - 1)$],), {
    Comment[Returns normalized node features]
    Comment[$b$th graph in the batch, number of nodes $N$, hidden dimension size $D$, @gnn layer $l$]
    State[]
    State[Sample @rnf $bold(R)_b^l in RR^(N times D)$]
    State[]
    Comment[Concatenate @rnf with $tilde(bold(H))_b^l in RR^(N times D)$ and pass through @granola's expressive @gnn]
    Assign[#v(0.5em) $bold(Z)_b^l$][$"GNN"_"Norm"^l (bold(A)_b, tilde(bold(H))_b^l || bold(Z)_b^l)$ #v(0.5em)]
    State[]
    Comment[Calculate affine parameters that are specific to the input graph]
    Assign[#v(0.5em) $gamma_(b, n)^l$][$f_1(z_(b, n)^l)$ #v(0.5em)]
    Assign[#v(0.5em) $beta_(b, n)^l$][$f_2(z_(b, n)^l)$ #v(0.5em)]
    State[]
    State[Compute mean $mu_(b, n, d)$ and standard deviation $sigma_(b, n, d)$ of $tilde(bold(H))_b^l in RR^(N times D)$ across the hidden dimension]
    State[]
    Return[$gamma_(b, n, d)^l (tilde(h)_(b, n, d)^l - mu_(b, n, d)) / sigma_(b, n, d) + beta_(b, n, d)^l$]
  })
}) <alg:granola>]
#place(top + center)[#granola]

Having motivated the need for input-specific affine parameters, we need a method of calculating them from the input graph---this is simply another @gnn. Since commonly used @gnn architectures are at most as powerful as the #box[@wl graph isomorphism heuristic @gin-paper], any normalization layer designed using them will be unable to distinguish all input graphs, and therefore will fail to adapt the normalization parameters correctly to suit the input.

More expressive architectures such as $k$-@gnn:pl @kgnn-paper, whose design is motivated by the generalization of 1-@wl to $k$−tuples of nodes ($k$-@wl), are accompanied by unacceptable computation and memory costs (e.g. $cal(O)(|V|^k)$ memory for higher-order @mpnn:pl, where $V$ is the number of nodes in the graph).

@rnf @rnf-paper is an easy to compute (and memory efficient), yet theoretically grounded alternative involving concatenating a different randomly generated vector to each node feature. This simple addition not only allows distinguishing between 1-@wl indistinguishable graph pairs based on fixed local substructures, but @gnn:pl augmented with @rnf are provably universal (with high probability), and thus can approximate any function defined on graphs of fixed order @rnf-power. In order to be maximally expressive, @granola uses an @mpnn @gnn-survey equipped with @rnf. It is important to note that @rnf breaks the invariance property of @gnn:pl, which is a strong inductive bias, however preserves it in expectation @rnf-power.

@granola facilitates an adaptive normalization layer by allowing its affine parameters $gamma_(b, n, d)^l, beta_(b, n, d)^l in RR$ to be dependent on the input-graph, by calculating them using a maximally expressive, shallow @gnn layer---$"GNN"_"Norm"$. A detailed overview of @granola is found in @alg:granola.

#modelexplanation[
  @granola has the potential to significantly improve the performance on this task as each overlap graph contains a unique set of artifacts (including none at all), so input-adaptivity is important. Additionally, prior work @lovro used BatchNorm/InstanceNorm, which the @granola authors show can lead to a loss in model capacity @granola-paper.
]

=== Complete model architecture
The complete model consists of three parts, shown in @fig:model_architecture. The first is an encoder layer, whose implementation is the feed-forward network described in @sec:standard_input_embedding, and converts the standard input features (@sec:standard_input_features) into the initial node and edge embeddings. The second part is the @gnn itself, that consists of 8 @gnn layers. If @granola is enabled, @granola layers are added after each @gnn layer too. The last part is the predictor layer that outputs the edge logits $e_(i j)^"logits"$ as follows:
$
  e_(i j)^"logits" = W_3 (thin #relu (W_2 (thin #relu (W_1 (h_i^"last" || h_j^"last" || e_(i j)^"last") + b_1)) + b_2)) + b_3
$
where all $W, b$ represent learnable parameters ($W_1 in RR^(D times 3D)$, $W_2 in RR^(32 times D)$, $W_3 in RR^(1 times 32)$, $b_1 in RR^D$, $b_2 in RR^32$, and $b_3 in RR$), and $D$ is the hidden dimension. $||$ denotes the concatenation operator. $h_i^"last"$, $h_j^"last"$ are the final-layer node embeddings, and $e_(i j)^"last"$ refers to the final-layer edge embeddings.

#place(top + center)[
  #figure(image("graphics/model_architecture.svg", height: 4cm), caption: [Illustration of the encoder-processor-decoder model architecture used. Note that the @gnn layer can be substituted with @symgatedgcn, GAT, or SymGAT.]) <fig:model_architecture>
]

== End-to-end neural genome assembly
In this project we also explore a wholly neural approach to genome assembly, where the @olc algorithm, along with overlap graph, is omitted. Allowing a neural network to define and control the entire assembly pipeline, rather than merely augmenting predefined stages, removes constraints and biases imposed by the traditional @olc framework.

Our architecture, @pgan is inspired by @ptrnet:pl @pointer-networks-paper that was designed to solve challenging geometric problems, including planar @tsp, making it a promising starting point for genome assembly. The general idea is that the model receives as input an unordered set of reads, and outputs a permutation of those reads (auto-regressively at test time), corresponding to the assembly.

Although this task superficially resembles a @s2s problem, conventional @s2s models are unsuitable since the output token "vocabulary" (the indices of the input reads) is dependent on the length of the input. This problem of the number of target classes in each step of the output depending on input length is handled by @ptrnet:pl through the use of the attention mechanism over each element (read) of the input. The element with the maximum attention score is the next token. Note that this approach also permits parallel training/teacher forcing in the same manner as training a decoder-only transformer.

@fig:neural-genome-assembly-custom-model illustrates our architecture, which is split into an encoder and decoder. We begin by explaining the encoder. Assume we start with a set of $N$ reads, ${r_i | i in {1, 2, .., N}}$ (in #text(fill: blue.lighten(20%))[blue]), forming an unordered set. These are encoded _individually_, by either Mamba, or an Encoder-only Transformer @transformer-paper (both represented by *R*), to create a read embedding $e_i$ for each read $r_i$ (in #text(fill: orange)[orange]). Note that we also add a special @eos embedding, $e_(N + 1)$.

In the decoder, we perform auto-regressive decoding. We start with the embedding of the read sequence ordered so far (in #text(fill: purple)[purple]) (prepended with a @bos embedding, $e_0$), $S = (e_0, e_a, e_b, ..., e_x)$ (where $a, b, ..., x$ represents some ordering of reads), and pass $S$ through another Mamba/Transformer sequence decoder. This produces encoded read sequence embeddings $D = (h_0, h_a, h_b, ..., h_x)$ (in #text(fill: red.lighten(25%))[pink]), which are passed through a pooling layer producing a hidden representation of the assembled sequence so far, $d$, by concatenating $"mean"(D)$ with the final embedding $h_x$.

Inspired by @ptrnet, the attention mechanism is then used to get attention scores of $d$ over all encoded read embeddings $e_i$. Additionally, we compute an overlap similarity score of each $e_i$ with $d$, and integrate this information into a separate attention head. Attention scores across all heads are then aggregated, with the next read in the sequence (dotted #text(fill: purple)[purple] arrow) decided by the $e_i$ with maximal attention.

Apart from the use of overlap similarity as additional information integrated into a separate attention head, our approach also differs from @ptrnet:pl in the encoder. We respect the unordered-set nature of the input collection of reads, and thus do not apply an @rnn like @ptrnet, as that breaks permutation equivariance of the representations $e_i$ produced.

#place(top + center)[
  #v(-2em)
  #subpar.grid(
    columns: (0.65fr, 1fr),
    show-sub-caption: sub-caption-styling,
    figure([#image("graphics/ptr-net.png") #v(0.5cm)], caption: [@ptrnet]), <fig:ptr-net>,
    figure(image("graphics/neural-genome-assembly.svg", height: 6cm), caption: [@pgan]), <fig:neural-genome-assembly-custom-model>,
    caption: [@pgan differs from @ptrnet is two key ways. First, we ensure to preserve the permutation equivariant nature of the encoder by not using an @rnn, and instead encoding each read individually. Second, we encode additional overlap information into the attention mechanism by exploiting attention heads.]
  )
]




#pagebreak()

= Evaluation and discussion
In this section, we present and discuss the results of five experiments. We:

+ Investigate whether alternative @gnn architectures (@gatedge and @symgatedge) can outperform the baseline @symgatedgcn @lovro on the original task of identifying erroneous edges in overlap graphs consisting solely of @pacbio @hifi long-read data.

+ Explore if the integration of ultra-long reads (from @ont @ul) helps in improving assembly quality by both more effectively detecting erroneous edges, and its impact on assembly contiguity.

+ Examine if @granola can help improve performance, especially during inference where the input overlap graphs are much larger than the partitioned subgraphs used during training.

+ Analyze Mamba's potential in extracting richer features directly from raw nucleotide-level read data, and improving model performance.

+ Survey the feasibility of end-to-end neural genome assembly, by testing @pgan on a simplified genome assembly task.

== Training, validation, and testing datasets <sec:datasets>
@fig:dataset_summary provides details for the (maternal) chromosomes (and specifically the regions within them) used as the training, validation, and testing datasets. 

The real-life overlap graphs for chromosomes chosen for both training and evaluation often consist of complex tangles, and are thus the most challenging to assemble. An acrocentric chromosome is one where the centromere, the region of a chromosome that holds sister chromatids together, is not located centrally on the chromosome, but towards one end. We utilize both non-acrocentric, and acrocentric chromosomes, as acrocentric chromosomes contain @rdna arrays @rdna-acrocentric harboring long tandem repeats that lead to distinct graph artifacts.

Furthermore, the centromeric region of each of these chromosomes is extracted for generating reads, where most assembly complexity arises @chm13-acrocentric. By training on only a small portion of the chromosomes present in the genome, we showcase the positive generalization capabilities of the @gnn\-based assembly paradigm.

#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => if y == 0 {
    (top: 0.7pt + black, bottom: 0.7pt + black)
  },
  align: (x, y) => { center }
)
#show table.cell: set align(horizon)
#show table.cell: set text(size: 0.85em)

#let yes_sym = [✅]
#let no_sym = [❌]

#let highlight = it => table.cell(fill: gray.lighten(65%))[#it]

#place(top + center)[#figure(
  table(
    columns: (7),
    table.header([Chr.], [Acrocentric], [Use], [Centromeric #linebreak() region start], [Centromeric #linebreak() region end], [Centromeric #linebreak() region len.], [Chr. length]),
    [19], [#no_sym], [Training], [19.8], [30.1], [10.3], [61.7],
    [15], [#yes_sym], [Training], [5.7], [15.5], [9.8], [99.8],
    highlight[11], highlight[#no_sym], highlight[Validation], highlight[48.7], highlight[56.2], highlight[7.5], highlight[135.1],
    highlight[22], highlight[#yes_sym], highlight[Validation], highlight[8.1], highlight[26.1], highlight[18.0], highlight[51.3],
    [9], [#no_sym], [Testing], [38.5], [75.4], [36.9], [150.6],
    [21], [#yes_sym], [Testing], [7.9], [14.7], [6.8], [45.1],
  ),
  caption: [Summary of the training, validation, and testing datasets used. Note that all data is from the maternal side.]
) <fig:dataset_summary>]

== Model performance and assembly quality metrics
We use standard metrics: *accuracy*, *precision*, *recall*, and *F1 score*,  to evaluate the model's performance on predicting erroneous edges. These are accompanied by their *_Inverse_* versions, which are calculated by setting the erroneous edge class as the "positive" class. We report both, as there are vastly fewer erroneous edges in the overlap graphs, and some observations and comparisons are made more salient under the *_Inverse_* version.

Additionally, we also utilize a number of commonly used metrics to assess the quality of the final genome assembly produced:

- *Number of contigs*: allows understanding how fragmented the genome assembly is. _Lower is better_.
- *Longest contig length*: long contigs indicate lower fragmentation too. _Higher is better_.
- *Genome fraction*: the percentage of the reference genome reconstructed by the assembly. _Higher is better_.
- *NG50*: this contiguity metric is computed by first sorting contigs by length (longest to shortest), and then taking the cumulative sum of those lengths, until the sum is $>50%$ of the reference genome length. The length of the contig at the $50%$ threshold is the result of this metric. _Higher is better_.
- *Number of mismatches*: the mean number of times where the nucleotide on the reference is different to that in the assembly, per $100$ @kb. _Lower is better_.
- *Number of indels*: the mean number of times where the assembly has either a nucleotide insertion or deletion compared to the reference, per $100$ @kb. _Lower is better_.

== Performance of alternative GNN layers <sec:performance_alt_gnn_layers>
Alternative @gnn layers fail to significantly and consistently outperform the baseline @symgatedgcn model on validation overlap graphs built solely using @pacbio @hifi long-reads. We see from @fig:similar_validation_performance that all three models: @symgatedgcn, @gatedge, and @symgatedge exhibit comparable performance, with overlapping $95%$ confidence intervals. All models achieve accuracy $>80%$, with a false positive rate of $~30%$, and a false negative rate of $~10%$, across both validation chromosomes 11 and 22.

We postulate that the underlying reason for this performance parity likely lies in the shared expressivity limitations of these architectures. None of the architectures exceed the graph distinguishing power of the (directed) 1-@wl test, and so share the same expressivity upper bound. Consequently, all models tested exhibit the same limitations regarding detection of local subgraph structures.

#place(top + center)[
*@symgatedgcn, @gatedge, and @symgatedge perform similarly*
#subpar.grid(
  columns: 3,
  column-gutter: -1em,
  show-sub-caption: sub-caption-styling,
  figure(image("graphics/base/key=validation_acc_epoch_train=19_valid=11_data=chm13htert-data_nodes=2000.png"), caption: [Validation Accuracy #linebreak() (Chromosome 11)]),
  figure(image("graphics/base/key=validation_fp_rate_epoch_train=19_valid=11_data=chm13htert-data_nodes=2000.png"), caption: [Validation False Positive #linebreak() (Chromosome 11)]),
  figure(image("graphics/base/key=validation_fn_rate_epoch_train=19_valid=11_data=chm13htert-data_nodes=2000.png"), caption: [Validation False Negative #linebreak() (Chromosome 11)]),
  figure(image("graphics/base/key=validation_acc_epoch_train=15_valid=22_data=chm13htert-data_nodes=2000.png"), caption: [Validation Accuracy #linebreak() (Chromosome 22)]),
  figure(image("graphics/base/key=validation_fp_rate_epoch_train=15_valid=22_data=chm13htert-data_nodes=2000.png"), caption: [Validation False Positive #linebreak() (Chromosome 22)]),
  figure(image("graphics/base/key=validation_fn_rate_epoch_train=15_valid=22_data=chm13htert-data_nodes=2000.png"), caption: [Validation False Negative #linebreak() (Chromosome 22)]),
  caption: [All three models (#text(fill: blue.darken(25%))[@symgatedgcn], #text(fill: orange)[@gatedge], #text(fill: green.darken(25%))[@symgatedge]) tested on overlap graphs generated solely using @pacbio @hifi reads perform similarly across both chromosomes 11 and 22 (trained on chromosomes 9, and 15 respectively). The darker line indicates the mean across $5$ runs, with the highlighted region indicating a $95%$ confidence interval.],
  label: <fig:similar_validation_performance>
)]

// Regarding global graph structure, exploring long-range dependency modelling in overlap graphs may be more important that initially thought. For example, by propagating substructure information from one bubble end-point to the other, the ambiguity regarding correct path selection in the bubble could be resolved. Additionally, such long-range information may make substructure detection easier.

One of the key features of @gat is the implicit ability of the model to assign different importances to nodes of the same neighborhood @gat-paper. We initially hypothesized that different types of graph artifacts would correspond to distinct substructures, requiring the model to selectively focus on different subsets of nodes and edges, playing to the strengths of the @gat architecture. Surprisingly, we empirically find this assumption to either be entirely false, or the weak expressive power of the networks preventing the model from identifying relevant features in the first place.

Furthermore, we find evidence that the @gat\-based architectures are over-fitting their training data. The baseline @symgatedgcn architecture convincingly outperforms the alternatives on the chromosome 9 test set (@tab:similar_test_performance shows significantly higher inverse precision, recall, and F1 score). This test set is significantly larger than the training overlap graph ($~3 #h(0em) times$ the size) (@fig:dataset_summary), and thus contains additional diversity in graph artifacts that can reveal over-fitting behavior.
The increased model capacity brought by the attention mechanism, in comparison to convolution, makes the model vulnerable to over-fitting. We believe that @gatedge's and @symgatedge's observed loss in performance on the test set is due to over-fitting as all three architectures have comparable performance on the smaller, and less diverse, training and validation datasets (@fig:similar_validation_performance).

Besides the previously discussed issues, another peculiarity we observe is that @gatedge and @symgatedge exhibit almost identical performance on both the validation (@fig:similar_validation_performance) and test (@tab:similar_test_performance) sets, when we expect @symgatedge to perform significantly better. The symmetry mechanism incorporates message passing in both the forward and reverse direction of the edges, in a manner that permits the model to distinguish the directionality of the information flow (which would not be possible via message passing on undirected edges). This symmetry mechanism, originally from the @dirgnn framework @dir-gnn-paper, provably makes the network equivalent in power to the _directed_ @wl test---strictly more expressive than standard @mpnn:pl.

Since the symmetry mechanism is particularly relevant for genome assembly, which requires finding a directed path through the overlap graph, and prior work @lovro has demonstrated the efficacy of the symmetry mechanism, we find these results unexpected. This phenomenon likely points to another more fundamental bottleneck limiting the performance of the @gat\-based architectures.

#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => if y == 0 {
    (top: 0.7pt + black, bottom: 0.7pt + black)
  },
  align: (x, y) => (
    if x > 0 { center }
    else { left }
  )
)

#let a_sd(accuracy, standard_deviation) = [#strfmt("{:.1}", accuracy * 100) $plus.minus$ #strfmt("{:.2}", standard_deviation * 100)]
#let best = it => [#strong(it)]

// Chr21:
// SymGatedGCN:

// acc: (0.9586893524716569, 0.008064994134126703), precision: (0.9995165691897043, 0.0002135035253943173), recall: (0.9580870593143731, 0.008378090132706588), f1: (0.9783480385671969, 0.004330907286733966)
// acc_inv: (0.9586893524716569, 0.008064994134126703), precision_inv: (0.38122619578172867, 0.043284155054646826), recall_inv: (0.982020202020202, 0.0080018590047208), f1_inv: (0.5479973168964583, 0.045259191527915035)

// GAT:

// acc: (0.9360775474927554, 0.00794433727919651), precision: (0.9954341333421655, 0.002905903316455245), recall: (0.9387612127112162, 0.010494042839725614), f1: (0.9662306803123208, 0.0044002242052899754)
// acc_inv: (0.9360775474927554, 0.00794433727919651), precision_inv: (0.261096448409076, 0.01873228841936153), recall_inv: (0.8321212121212122, 0.10845242761164156), f1_inv: (0.39590827636523246, 0.021421343092625816)

// SymGAT:

// acc: (0.9648477350912572, 0.0067034830095140064), precision: (0.9990626647918466, 0.0003915836978962343), recall: (0.9648459773312009, 0.006933219132265786), f1: (0.9816458693278843, 0.0035733402659254444)
// acc_inv: (0.9648477350912572, 0.0067034830095140064), precision_inv: (0.41878887677930976, 0.044617504904132695), recall_inv: (0.964915824915825, 0.014721546014621454), f1_inv: (0.5829126678063014, 0.04427662087367158)

#let highlight = it => table.cell(fill: yellow.lighten(50%))[#it]
#let highlighted = [(#box(fill: yellow.lighten(50%))[highlighted])]

#place(top + center)[#figure(table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header([Test Metric (%)], [@symgatedgcn], [@gatedge], [@symgatedge]),
  [$arrow.t$ Accuracy], best[#a_sd(0.8595862227139041, 0.0036745137412503687)], [#a_sd(0.852172431936537, 0.018122915137703973)], [#a_sd(0.8420853457812795, 0.017519697090493506)],
  [$arrow.t$ Precision], best[#a_sd(0.9634490911519177, 0.0025629919337929147)], [#a_sd(0.9491644376556241, 0.0010797849002164902)], [#a_sd(0.9526119974753542, 0.001415315356238136)],
  [$arrow.t$ Recall], [#a_sd(0.8759740235208783, 0.006283861818846527)], best[#a_sd(0.8816506171113823, 0.021949458751530884)], [#a_sd(0.8662298215722646, 0.02142411761762067)],
  [$arrow.t$ F1], best[#a_sd(0.9176147356522699, 0.002512762417627877)], [#a_sd(0.914044335828298, 0.011599446124280127)], [#a_sd(0.9072501961743051, 0.011465972986112126)],
  highlight[$arrow.t$ Precision Inverse], highlight(best[#a_sd(0.4119825030501988, 0.006363473984693595)]), highlight[#a_sd(0.3848171812904071, 0.038358039650424273)], highlight[#a_sd(0.36794115048395104, 0.030808877124403163)],
  highlight[$arrow.t$ Recall Inverse], highlight(best[#a_sd(0.7231195142668386, 0.0218981435468089)]), highlight[#a_sd(0.6066977066096203, 0.015938260139693062)], highlight[#a_sd(0.641026205681568, 0.017880694079863063)],
  highlight[$arrow.t$ F1 Inverse], highlight(best[#a_sd(0.5247472552791161, 0.0053616664852087735)]), highlight[#a_sd(0.46957065155016847, 0.02481152760711465)], highlight[#a_sd(0.4665307562943479, 0.02181144060394172)],
),
  caption: [The baseline @symgatedgcn outperforms both @gatedge and @symgatedge when the models are trained on chromosome 15, and tested on chromosome 9, with much higher inverse precision, recall, and F1 score #highlighted. The results show the mean and standard deviation across 5 runs. Best results in *bold*. $arrow.t$ indicates _higher is better_.]
) <tab:similar_test_performance>]

== Integration of ultra-long data <sec:integration-ultra-long-data>
#place(top + center)[
*@symgatedgcn's performance improves substantially with ultra-long reads*
#subpar.grid(
  columns: 3,
  column-gutter: -1em,
  show-sub-caption: sub-caption-styling,
  figure(image("graphics/ul/key=validation_acc_epoch_train=19_valid=11_nodes=2000.png"), caption: [Validation Accuracy #linebreak() (Chromosome 11)]), <subfig:better_acc_11>,
  figure(image("graphics/ul/key=validation_fp_rate_epoch_train=19_valid=11_nodes=2000.png"), caption: [Validation False Positive #linebreak() (Chromosome 11)]), <subfig:better_fp_11>,
  figure(image("graphics/ul/key=validation_fn_rate_epoch_train=19_valid=11_nodes=2000.png"), caption: [Validation False Negative #linebreak() (Chromosome 11)]), <subfig:better_fn_11>,
  figure(image("graphics/ul/key=validation_acc_epoch_train=15_valid=22_nodes=2000.png"), caption: [Validation Accuracy #linebreak() (Chromosome 22)]), <subfig:better_acc_22>,
  figure(image("graphics/ul/key=validation_fp_rate_epoch_train=15_valid=22_nodes=2000.png"), caption: [Validation False Positive #linebreak() (Chromosome 22)]), <subfig:better_fp_22>,
  figure(image("graphics/ul/key=validation_fn_rate_epoch_train=15_valid=22_nodes=2000.png"), caption: [Validation False Negative #linebreak() (Chromosome 22)]), <subfig:better_fn_22>,
  caption: [@symgatedgcn trained and validated on overlap graphs generated with additional @ont @ul reads (@symgatedgcn (@ul) in #text(fill: orange)[orange]) outperforms @symgatedgcn trained and validated on overlap graphs generated solely using @pacbio @hifi reads (in #text(fill: blue.darken(25%))[blue]) across both chromosomes 11 and 22. The darker line indicates the mean across $5$ runs, with the highlighted region indicating a $95%$ confidence interval.]
)]

All three architectures tested, @symgatedgcn, @gatedge, and @symgatedge, offer substantial performance gains when operating on overlap graphs augmented with ultra-long data. For example, on validation chromosome 11, we see that @symgatedgcn achieves higher accuracy (@subfig:better_acc_11), fewer false negatives (@subfig:better_fn_11) and far fewer false positives (@subfig:better_fp_11) compared to its performance on overlap graphs lacking ultra-long data. We observe a similar trend with validation chromosome 22---substantial decrease in false positives (@subfig:better_fp_22), although the accuracy (@subfig:better_acc_22) and false negatives exhibit limited change, due to already excellent baseline performance (@subfig:better_fn_22).

This success extends to the test set where all three models see a significant uplift to the inverse precision and inverse F1 score, with negligible impact on other metrics such as accuracy. For example, with @symgatedgcn, ultra-long read data helps to improve inverse precision from $41.2%$ to $52.3%$, and the inverse F1 score from $52.5%$ to $60.4%$ (@tab:symgatedgcn_test). The complete suite of data can be found @app:chr-19-ul-granola-test, which shows analogous performance improvements for @gatedge and @symgatedge. These findings provide compelling evidence that all tested @gnn:pl are successfully able to leverage ultra-long data to improve erroneous edge detection in overlap graphs.

Moreover, ultra-long data aids in improving assembly quality. Taking the @symgatedgcn model as an example again, @tab:symgatedgcn_assembly shows that ultra-long data helps cover an increased genome fraction of the reference assembly from an average of $92.7%$ to $93.6%$. The maximum coverage achieved with ultra-long reads was $95.1%$, compared to $93.9%$ without. Although these results may seem like trivial improvements, they hold significant value for achieving @t2t assemblies, as it is precisely these small, complex regions of the genome that were previously omitted from assemblies, leading to gaps and fragmentation.

Furthermore, this greater coverage is accomplished without any significant detrimental impact on assembly quality. @symgatedgcn's assembly with ultra-long data achieves a similar mismatch and indel rate, with only a slight reduction in the length of the longest contig and NG50 (@tab:symgatedgcn_assembly). In fact, the total number of contigs decreases, which is an indicator of a less fragmented assembly. We observe similar benefits to assembly quality from ultra-long reads with @gatedge and @symgatedge models too (@app:chr-19-ul-granola-assembly).


// #place(top + center)[#figure(table(
//   columns: (auto, 1fr, 1fr, 1fr),
//   table.header([Test Metric (%)], [@symgatedgcn], [GAT+Edge], [SymGAT+Edge]),
//   [Accuracy], [#a_sd(0.9586893524716569, 0.008064994134126703)], [#a_sd(0.9360775474927554, 0.00794433727919651)], best[#a_sd(0.9648477350912572, 0.0067034830095140064)],
//   [Precision], best[#a_sd(0.9995165691897043, 0.0002135035253943173)], [#a_sd(0.9954341333421655, 0.002905903316455245)], [#a_sd(0.9990626647918466, 0.0003915836978962343)],
//   [Recall], [#a_sd(0.9580870593143731, 0.008378090132706588)], [#a_sd(0.9387612127112162, 0.010494042839725614)], best[#a_sd(0.9648459773312009, 0.006933219132265786)],
//   [F1], [#a_sd(0.9783480385671969, 0.004330907286733966)], [#a_sd(0.9662306803123208, 0.0044002242052899754)], best[#a_sd(0.9816458693278843, 0.0035733402659254444)],
//   highlight[Precision Inverse], highlight[#a_sd(0.38122619578172867, 0.043284155054646826)], highlight[#a_sd(0.261096448409076, 0.01873228841936153)], highlight(best[#a_sd(0.41878887677930976, 0.044617504904132695)]),
//   [Recall Inverse], best[#a_sd(0.982020202020202, 0.0080018590047208)], [#a_sd(0.8321212121212122, 0.10845242761164156)], [#a_sd(0.964915824915825, 0.014721546014621454)],
//   highlight[F1 Inverse], highlight[#a_sd(0.5479973168964583, 0.045259191527915035)], highlight[#a_sd(0.39590827636523246, 0.021421343092625816)], highlight(best[#a_sd(0.5829126678063014, 0.04427662087367158)]),
// ),
//   caption: [SymGAT+Edge outperforms both @symgatedgcn and GAT+Edge when the models are trained on chromosome 15, and tested on chromosome 21. The results show the mean and standard deviation across 5 runs.]
// ) <tab:similar_test_performance>]


#let mean(array) = array.sum() / array.len()
#let std(array) = calc.sqrt(mean(array.map(v => calc.pow(v - mean(array), 2))))
#let a_sd_a(array, multiplier: 1) = [#strfmt("{:.1}", mean(array) * multiplier) $plus.minus$ #strfmt("{:.2}", std(array) * multiplier)]
// [?]
// #table(
//   columns: (auto, 1fr, 1fr, 1fr),
//   table.header([Assembly Metric], [@symgatedgcn], [GAT+Edge], [SymGAT+Edge]),
//   [Num. contigs], [#a_sd_a(base-chr9-SymGatedGCN-contigs)], best[#a_sd_a(base-chr9-GAT-contigs)], [#a_sd_a(base-chr9-SymGAT-contigs)],
//   [Longest contig length], [#a_sd_a(base-chr9-SymGatedGCN-largest-contig, multiplier: 0.0000001)], best[#a_sd_a(base-chr9-GAT-largest-contig, multiplier: 0.0000001)], [#a_sd_a(base-chr9-SymGAT-largest-contig, multiplier: 0.0000001)],
//   [Genome fraction (%)], best[#a_sd_a(base-chr9-SymGatedGCN-genome-fraction)], [#a_sd_a(base-chr9-GAT-genome-fraction)], [#a_sd_a(base-chr9-SymGAT-genome-fraction)],
//   [NG50], [#a_sd_a(base-chr9-SymGatedGCN-ng50, multiplier: 0.0000001)], best[#a_sd_a(base-chr9-GAT-ng50, multiplier: 0.0000001)], [#a_sd_a(base-chr9-SymGAT-ng50, multiplier: 0.0000001)],
//   [NGA50], [#a_sd_a(base-chr9-SymGatedGCN-nga50, multiplier: 0.0000001)], [#a_sd_a(base-chr9-GAT-nga50, multiplier: 0.0000001)], best[#a_sd_a(base-chr9-SymGAT-nga50, multiplier: 0.0000001)],
//   [Num. mismatches (per 100 @kb)], [#a_sd_a(base-chr9-SymGatedGCN-mismatches)], best[#a_sd_a(base-chr9-GAT-mismatches)], [#a_sd_a(base-chr9-SymGAT-mismatches)],
//   [Num. indels (per 100 @kb)], best[#a_sd_a(base-chr9-SymGatedGCN-indels)], [#a_sd_a(base-chr9-GAT-indels)], [#a_sd_a(base-chr9-SymGAT-indels)],
// )

// #show table.cell.where(y: 5): it => {
//   table.cell(
//     fill: gray,
//     strong(it)
//   )
// }
// #show table.cell.where(y: 5): it => {table.cell(fill: yellow)[#it]}




#place(top + center)[#figure(
table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header([Metric (%)], [@symgatedgcn], [@symgatedgcn (@ul)], [@symgatedgcn (@ul+@granola)]),
  [$arrow.t$ Accuracy], best[#a_sd(0.8595862227139041, 0.0036745137412503687)], [#a_sd(0.8321690738555465, 0.0025772443038743565)], [#a_sd(0.8301424922853613, 0.007101224726230073)],
  [$arrow.t$ Precision], best[#a_sd(0.9634490911519177, 0.0025629919337929147)], [#a_sd(0.9323424023725854, 0.004516353276930694)], [#a_sd(0.9160111158239687, 0.00972272749589809)],
  [$arrow.t$ Recall], best[#a_sd(0.8759740235208783, 0.006283861818846527)], [#a_sd(0.8578891257995735, 0.0063061462302990414)], [#a_sd(0.8734835092170252, 0.020028524044763716)],
  [$arrow.t$ F1], best[#a_sd(0.9176147356522699, 0.002512762417627877)], [#a_sd(0.8935430005291409, 0.0019810980223524005)], [#a_sd(0.8940413062979149, 0.0061798984540991165)],
  highlight[$arrow.t$ Precision Inverse], highlight[#a_sd(0.4119825030501988, 0.006363473984693595)], highlight(best[#a_sd(0.5228007379159743, 0.005925397365242645)]), highlight[#a_sd(0.522541581304234, 0.017236383412959296)],
  [$arrow.t$ Recall Inverse], best[#a_sd(0.7231195142668386, 0.0218981435468089)], [#a_sd(0.7141452766510822, 0.02227860874135133)], [#a_sd(0.6312598693377094, 0.05396491671958023)],
  highlight[$arrow.t$ F1 Inverse], highlight[#a_sd(0.5247472552791161, 0.0053616664852087735)], highlight(best[#a_sd(0.6035159891309024, 0.007614321070066256)]), highlight[#a_sd(0.5702396833678883, 0.012031478690339393)],
),
caption: [There is a significant performance uplift when ultra-long data is integrated into the overlap graph (@symgatedgcn (@ul)), with much higher inverse precision and F1 score #highlighted. @granola does not help in improving performance further. The results show the mean and standard deviation across 5 runs, with chromosome 15 used for training, and 9 as the testing dataset for these metrics. Best results in *bold*. $arrow.t$ indicates _higher is better_.]
) <tab:symgatedgcn_test>
#figure(
table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header([Assembly Metric], [@symgatedgcn], [@symgatedgcn (@ul)], [@symgatedgcn (@ul+@granola)]),
  [$arrow.b$ Num. contigs], [#a_sd_a(base-chr9-SymGatedGCN-contigs)], best[#a_sd_a(ul-chr9-SymGatedGCN-contigs)], [#a_sd_a(granola-ul-chr9-SymGatedGCN-contigs)],
  [$arrow.t$ Longest contig length], best[#a_sd_a(base-chr9-SymGatedGCN-largest-contig, multiplier: 0.0000001)], [#a_sd_a(ul-chr9-SymGatedGCN-largest-contig, multiplier: 0.0000001)], [#a_sd_a(granola-ul-chr9-SymGatedGCN-largest-contig, multiplier: 0.0000001)],
  highlight[$arrow.t$ Genome fraction (%)], highlight[#a_sd_a(base-chr9-SymGatedGCN-genome-fraction)], highlight(best[#a_sd_a(ul-chr9-SymGatedGCN-genome-fraction)]), highlight[#a_sd_a(granola-ul-chr9-SymGatedGCN-genome-fraction)],
  [$arrow.t$ NG50], best[#a_sd_a(base-chr9-SymGatedGCN-ng50, multiplier: 0.0000001)], [#a_sd_a(ul-chr9-SymGatedGCN-ng50, multiplier: 0.0000001)], [#a_sd_a(granola-ul-chr9-SymGatedGCN-ng50, multiplier: 0.0000001)],
  // [NGA50], [#a_sd_a(base-chr9-SymGatedGCN-nga50, multiplier: 0.0000001)], [#a_sd_a(ul-chr9-SymGatedGCN-nga50, multiplier: 0.0000001)], best[#a_sd_a(granola-ul-chr9-SymGatedGCN-nga50, multiplier: 0.0000001)],
  [$arrow.b$ Num. misma. (per 100 @kb)], [#a_sd_a(base-chr9-SymGatedGCN-mismatches)], [#a_sd_a(ul-chr9-SymGatedGCN-mismatches)], best[#a_sd_a(granola-ul-chr9-SymGatedGCN-mismatches)],
  [$arrow.b$ Num. indels (per 100 @kb)], best[#a_sd_a(base-chr9-SymGatedGCN-indels)], [#a_sd_a(ul-chr9-SymGatedGCN-indels)], [#a_sd_a(granola-ul-chr9-SymGatedGCN-indels)],
),
caption: [Integration of ultra-long data into the overlap graph (@symgatedgcn (@ul)) results in a higher fraction of the reference genome being covered in the reconstructed assembly #highlighted. This is achieved whilst maintaining assembly quality. @granola is detrimental to assembly quality. The results show the mean and standard deviation across 5 runs, with chromosome 15 used for training, and 9 as the reference genome assembled. Best results in *bold*. $arrow.t$ indicates _higher is better_. $arrow.b$ indicates _lower is better_.]
) <tab:symgatedgcn_assembly>
]

== Performance impact of GRANOLA <sec:granola-performance>
@granola fails to provide any consequential performance improvement on any of the @gnn layers tested. During validation on chromosome 11, @granola increases the number of false positives, and on chromosome 22, it causes a slight reduction in accuracy, and an increase in false negatives. Furthermore, from @tab:symgatedgcn_test, we find that the addition of @granola is slightly detrimental to overall model performance on the chromosome 9 test set, with a lower inverse F1 Score, and significantly lower inverse recall in comparison to @symgatedgcn with ultra-long data. This is unexpected as @granola, on the surface, seems beneficial for this problem due to three reasons.

Firstly, we postulated that  graph adaptability would be beneficial when testing on overlap graphs from different chromosomes as the distribution of graph artifacts, and overall graph size changes.

Moreover, @granola uses maximally expressive @gnn layers to adapt the normalization parameters to the graph, and model expressivity seems crucial for this task, as discussed earlier in @sec:performance_alt_gnn_layers.

Lastly, all models not utilizing @granola utilize InstanceNorm @instancenorm, which the @granola authors show can limit model capacity, for example by preventing the model from computing critical features such as node degrees @granola-paper. Judging from the results we observed, the model capacity limitation from the use of InstanceNorm is not a practical concern, and the adaptability afforded by @granola likely induces over-fitting, as it performed more on-par with the baseline during training.

@tab:symgatedgcn_assembly shows that @granola's disappointing performance is also reflected in the quality of the final assembly produced, where the increased genome coverage achieved by integration of long-reads is erased. This perhaps also signals that @granola is not as well suited to integrating long-read information. Additional data regarding @granola's performance on the other architectures can be found in @app:chr-19-ul-granola-assembly.

#place(top + center)[
  *Addition of @granola causes a performance regression with #linebreak() @symgatedgcn + ultra-long reads*
  #subpar.grid(
    columns: 3,
    column-gutter: -1em,
    show-sub-caption: sub-caption-styling,
    figure(image("graphics/granola-ul/key=validation_acc_epoch_train=19_valid=11_nodes=2000.png"), caption: [Validation Accuracy #linebreak() (Chromosome 11)]),
    figure(image("graphics/granola-ul/key=validation_fp_rate_epoch_train=19_valid=11_nodes=2000.png"), caption: [Validation False Positive #linebreak() (Chromosome 11)]),
    figure(image("graphics/granola-ul/key=validation_fn_rate_epoch_train=19_valid=11_nodes=2000.png"), caption: [Validation False Negative #linebreak() (Chromosome 11)]),
    figure(image("graphics/granola-ul/key=validation_acc_epoch_train=15_valid=22_nodes=2000.png"), caption: [Validation Accuracy #linebreak() (Chromosome 22)]),
    figure(image("graphics/granola-ul/key=validation_fp_rate_epoch_train=15_valid=22_nodes=2000.png"), caption: [Validation False Positive #linebreak() (Chromosome 22)]),
    figure(image("graphics/granola-ul/key=validation_fn_rate_epoch_train=15_valid=22_nodes=2000.png"), caption: [Validation False Negative #linebreak() (Chromosome 22)]),
    caption: [The addition of @granola (@symgatedgcn (@ul + @granola) in #text(fill: blue.darken(25%))[blue]) to @symgatedgcn trained and validated on overlap graphs generated with additional @ont @ul reads (@symgatedgcn (@ul) in #text(fill: orange)[orange]) diminishes performance across both chromosomes 11 and 22. The darker line indicates the mean across $5$ runs, with the highlighted region indicating a $95%$ confidence interval.]
  )
]

== Mamba's potential for richer feature extraction <sec:mamba-potential-feature-extraction>
Mamba shows capability in eliciting useful features from raw nucleotide read data for the genome assembly task. For example, on the chromosome 21 test set, @symgatedgcn-mambaedge achieves significantly higher inverse precision and F1 score than the baseline @symgatedgcn model (@tab:mamba-test). In particular, @symgatedgcn-mambaedge outperforms @symgatedgcn-randomedge, which in @sec:symgatedgcn-randomedge, is established as the baseline for Mamba's performance, since a randomly initialized Mamba model can be viewed as generating random edge features.

It is intriguing that @symgatedgcn-mamba performs rather poorly on this test set, achieving the lowest inverse F1 score across all models tested. We believe that this is the case due to @symgatedgcn-mamba having lower model capacity than @symgatedgcn-mambaedge. Recall from @eqn:symgatedgcn-update-hidden that @symgatedgcn updates the hidden state by taking a gated linear transformation of the previous hidden states, and from @eq:edge_features that the edge features are also updated using a linear transformation of the previous hidden states. These linear transformations limit model capacity compared to multi-layer feed-forward networks. Since @symgatedgcn-mamba uses Mamba to generate new _node_ features, despite these features being potentially richer, they might not be effectively used to update the hidden edge embedding, due to this capacity limitation.

In addition to this, the performance uplift of @symgatedgcn-randomedge over @symgatedgcn potentially demonstrates the utility of @rnf, which theoretically increases the expressive power of @gnn:pl @rnf-power.

However, despite the positive results observed with the introduction of both Mamba and the random edge features, we remain cautious regarding their efficacy. Due to compute and memory limitations, the input overlap graphs were heavily subsampled, and the read profile used to simulate reads was modified to generate shorter reads. This was required to compensate for Mamba's substantial resource requirements. Nevertheless, this promising result holds value for future work.

// Chr21 (trained on 15):
// SymGatedGCN:
// acc: (0.984650974597143, 0.00433200250243265), precision: (0.9998194483804524, 7.827066227169826e-05), recall: (0.9847941461694437, 0.004409870891316927), f1: (0.99224582346533, 0.002202133403092189)
// acc_inv: (0.984650974597143, 0.00433200250243265), precision_inv: (0.12864798479512254, 0.036629013492757476), recall_inv: (0.9217054263565891, 0.03438489165304943), f1_inv: (0.22378802267917886, 0.05295265051308235)


// Mamba1:
// acc: (0.9874513847221182, 0.004426172913764797), precision: (0.9997186789261707, 0.00024953321741186485), recall: (0.9876949336367117, 0.004381297563484567), f1: (0.9936658651173959, 0.0022527199653774757)
// acc_inv: (0.9874513847221182, 0.004426172913764797), precision_inv: (0.1285348236348254, 0.08098702091148237), recall_inv: (0.8425925520662363, 0.08589836722055305), f1_inv: (0.21500261576482105, 0.1209215833998852)

// Mamba2:
// acc: (0.9915768761235079, 0.003980587400663833), precision: (0.9995182993701981, 0.00045280664065934674), recall: (0.9920303347345625, 0.003868304557344345), f1: (0.9957567633400288, 0.0020165341111040625)
// acc_inv: (0.9915768761235079, 0.003980587400663833), precision_inv: (0.17015542857861995, 0.09585003337199395), recall_inv: (0.7835704125177809, 0.0775469993438432), f1_inv: (0.26710222164199865, 0.11682858910597882)

// Random:
// acc: (0.9886971360213919, 0.001913964595413109), precision: (0.9996775509604485, 9.71850643912885e-05), recall: (0.9889905668694349, 0.001998789898287203), f1: (0.9943044637899703, 0.0009702786100677676)
// acc_inv: (0.9886971360213919, 0.001913964595413109), precision_inv: (0.15316415826771998, 0.018481853937947625), recall_inv: (0.8596899224806202, 0.04258277199061171), f1_inv: (0.259293321499784, 0.02507388915237946)

#place(top + center)[
  #figure(table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    table.header([Metric (%)], [@symgatedgcn], [#gls("symgatedgcn-mamba", display: "SymGatedGCN +Mamba")], [#gls("symgatedgcn-mambaedge", display: "SymGatedGCN + MambaEdge")], [#gls("symgatedgcn-randomedge", display: "SymGatedGCN +RandomEdge")]),
    [$arrow.t$ Accuracy], [#a_sd(0.984650974597143, 0.00433200250243265)], [#a_sd(0.9874513847221182, 0.004426172913764797)], best[#a_sd(0.9915768761235079, 0.003980587400663833)], [#a_sd(0.9886971360213919, 0.001913964595413109)],
    [$arrow.t$ Precision], [#a_sd(0.9998194483804524, 7.827066227169826e-05)], best[#a_sd(0.9997186789261707, 0.00024953321741186485)], [#a_sd(0.9995182993701981, 0.00045280664065934674)], [#a_sd(0.9996775509604485, 9.71850643912885e-05)],
    [$arrow.t$ Recall], [#a_sd(0.9847941461694437, 0.004409870891316927)], [#a_sd(0.9876949336367117, 0.004381297563484567)], best[#a_sd(0.9920303347345625, 0.003868304557344345)], [#a_sd(0.9889905668694349, 0.001998789898287203)],
    [$arrow.t$ F1], [#a_sd(0.99224582346533, 0.002202133403092189)], [#a_sd(0.9936658651173959, 0.0022527199653774757)], best[#a_sd(0.9957567633400288, 0.0020165341111040625)], [#a_sd(0.9943044637899703, 0.0009702786100677676)],
    highlight[$arrow.t$ Precision Inv.], highlight[#a_sd(0.12864798479512254, 0.036629013492757476)], highlight[#a_sd(0.1285348236348254, 0.08098702091148237)], highlight(best[#a_sd(0.17015542857861995, 0.09585003337199395)]), highlight[#a_sd(0.15316415826771998, 0.018481853937947625)],
    [$arrow.t$ Recall Inv.], best[#a_sd(0.9217054263565891, 0.08589836722055305)], [#a_sd(0.8425925520662363, 0.08589836722055305)], [#a_sd(0.7835704125177809, 0.0775469993438432)], [#a_sd(0.8596899224806202, 0.04258277199061171)],
    highlight[$arrow.t$ F1 Inv.], highlight[#a_sd(0.22378802267917886, 0.05295265051308235)], highlight[#a_sd(0.21500261576482105, 0.1209215833998852)], highlight(best[#a_sd(0.26710222164199865, 0.11682858910597882)]), highlight[#a_sd(0.259293321499784, 0.02507388915237946)],
  ),
  caption: [@symgatedgcn-mambaedge outperforms the baseline @symgatedgcn and @symgatedgcn-randomedge models, achieving higher inverse precision and inverse F1 score #highlighted. The results show the mean and standard deviation across 5 runs, with chromosome 15 used for training, and 21 as the testing dataset for these metrics. Best results in *bold*. $arrow.t$ indicates _higher is better_.]
  ) <tab:mamba-test>
]

== Experimenting with end-to-end neural genome assembly
We demonstrate the feasibility of end-to-end neural genome assembly by showcasing the promising results of our @pgan architecture on a simplified version of the genome assembly problem, where all reads are perfect and of a fixed length. However, this problem is still challenging as the neural network needs to find the correct permutation of reads corresponding to the original reference.

=== Dataset generation
The _entirety_ of human chromosomes 19, 18, and 21, is utilized for training, validation, and testing respectively. For each chromosome, we prepare the dataset by partitioning it into contiguous regions 10 @kb long---each of these regions forms a minibatch. Then, within each region, we randomly sample reads with perfect accuracy. The reads are then permuted under a random permutation $P$, and the target is to predict the inverse permutation $P^(-1)$, such that $P P^(-1) = I = P^(-1) P$, where $I$ represents the identity permutation. Note that the minibatch reads and permutations are randomized every epoch.

=== Comparing Mamba and Transformer performance
#place(top + center)[
  *Mamba outperforms Transformer #linebreak() by avoiding overfitting to the training dataset*
  #subpar.grid(
    columns: 2,
    show-sub-caption: sub-caption-styling,
    figure(image("graphics/neural-genome-assembly/key=train_loss_train=19_valid=18.png"), caption: [Training Loss (Chromosome 19)]), <fig:mamba-transformer-training>,
    figure(image("graphics/neural-genome-assembly/key=validation_loss_train=19_valid=18.png"), caption: [Validation Loss (Chromosome 18)]), <fig:mamba-transformer-validation>,
    caption: [Mamba (in #text(fill: orange)[orange]) performs similarly to Transformer (in #text(fill: blue.darken(25%))[blue]) during training, but vastly outperforms it during validation. The darker line indicates the mean across $5$ runs, with the highlighted region indicating a $95%$ confidence interval.]
  )
]

We investigate two options for encoding reads, and decoding read sequences within the @pgan architecture---Mamba and Transformer. Although @fig:mamba-transformer-training shows that both models have comparable performance during training, we see diverging loss from the Transformer during validation (@fig:mamba-transformer-validation), while the Mamba version generalizes significantly better. This result extends to the chromosome 21 test set, where the Mamba-based model has an average of $72.4% (plus.minus 10.6%)$ accuracy, compared to Transformer's $21.8% (plus.minus 10.8%)$, across $5$ runs.

We theorize that this is a result of a lack of a canonical tokenization scheme for @dna that effectively generalizes to unseen, de novo sequences, in contrast to natural language. Consequently, both models are constrained to operate directly on raw nucleotide read data, where the Mamba model is particularly effective. Although there exist @dna tokenizers like DNABERT(2) @dnabert @dnabert-2, and entropy-based tokenization schemes such as the Byte Latent Transformer @byte-latent-transformer, we delegate their exploration to future work, as tokenization is not the focus of this project. Crucially, this result also reinforces our choice of using Mamba in earlier experiments discussed in this dissertation.





#pagebreak()

= Summary and conclusions
This chapter summaries the experiment results, and proposes directions for future work.
== Overview and insights gained
In summary, this project was successful in meeting all of its original aims. We now present each goal, the contributions made towards that goal, and insights gained in the process.

#let aim_achieved = it => [
  #box(fill: green.lighten(90%), inset: 1em, stroke: green, radius: 1em, width: 100%)[#it]
]

#aim_achieved[
  *Aim 1:*
  #aim_1
]

Two @gnn architectures extending the original @gat layers incorporating edge features were introduced---@gatedge, and the novel @symgatedge that integrated the symmetry mechanism found in @symgatedgcn. A series of experiments were run, which uncovered that these alternate architectures did not significantly outperform the prior @symgatedgcn baseline in predicting erroneous edges in overlap graphs, nor in improving assembly quality.

From the evidence gathered, we hypothesized that since none of these architectures exceeded the expressive power of the (directed) 1-@wl test, and because layout is fundamentally a structural problem, a noteworthy performance improvement should not be expected. Additionally, the design of any @gnn architecture for this task must be mindful of the possibility of over-fitting, induced by increased model capacity.

The alternative @granola normalization layer held potential for furthering performance due to its input graph adaptability, and the proven model capacity reduction that could be caused by the use of standard normalization schemes such as InstanceNorm. In testing, @granola proved ultimately unsuccessful, with its graph adaptability mechanism hindering performance. Moreover, the theoretical model capacity reduction did not seem consequential in practice, with the InstanceNorm baseline outperforming @granola.

#aim_achieved[
  *Aim 2:*
  #aim_2
]

Ultra-long data was assimilated into the overlap graph through `Hifiasm (UL)`. All @gnn architectures tested effectively utilized this additional information, providing substantial and consistent performance improvements. A critical metric was the expanded genome coverage, without significantly sacrificing assembly quality, helping in the achievement of @t2t assemblies. It also did not necessitate the use of more advanced @gnn layers, with @symgatedgcn outperforming @gatedge and @symgatedge. Moreover, the success of the ultra-long datatype holds promise for substituting the need for more expressive @gnn architectures.

#aim_achieved[
  *Aim 3:*
  #aim_3
]
Mamba was used to translate raw nucleotide reads into fixed-length embeddings. These were then used to either form node or edge features. Mamba generating edge features outperformed the baseline that used normalized overlap length and similarity as edge features. Furthermore, it also outperformed random edge features. Mamba generating node features failed to surpass the baseline, and even the random edge features, which is likely due to the model capacity constraint imposed by @symgatedgcn in using node features to update edge features.

#aim_achieved[
  *Aim 4:*
  #aim_4
]

We introduced a novel architecture, @pgan, which successfully demonstrated the feasibility of end-to-end neural genome assembly in a simplified (yet still challenging) scenario where reads are perfectly accurate. Two variants of the architecture were evaluated---one utilizing Mamba, and the other Transformer. The Mamba-based model performed significantly better, which we hypothesize is due to its ability to operate more effectively on raw nucleotide-level data, in the absence of any canonical tokenization scheme for de novo @dna sequences.

== Future work
This project reveals several compelling directions for future work. The first direction is a more concrete analysis of the importance of highly expressive @gnn:pl for layout in genome assembly. Even though higher expressivity architectures like $k$-@gnn:pl are impractical for general use, observing their performance on this task would provide an objective baseline for the comparing the use of more computationally efficient, expressive @gnn architectures, and reveal the extent to which expressiveness is critical for this task. 

Furthermore, the Graph Transformer architecture @graph-transformer can also be trialed on this problem. By not leveraging the graph connectivity inductive bias, performance may be improved especially across long distances in the graph---potentially particularly useful with ultra-long read data. This could also help surface and mitigate any #box[over-squashing @over-squashing] and under-reaching @under-reaching phenomenon encountered by any of the tested architectures. Additionally, while we currently focus our efforts on detecting and removing erroneous edges, Graph Transformer Networks could establish new connections between unconnected nodes on the original graph @graph-transformer-new-connections, helping improve assembly contiguity. Unfortunately, due to compute limitations, we were unable to experiment with this architecture for ourselves.

Moreover, recent work has substituted more expressive models and architectural inductive biases with additional data and regularization @alpha-fold-3. Notably, it has been empirically demonstrated @qualcomm-equivariance-scale that while equivariance improves data efficiency, given sufficient compute, data augmentation closes the performance gap. Exploring a data-driven path to improving performance is thus interesting.

@symgatedge's failure to thoroughly outperform its @gatedge counterpart was surprising, and so future work could perform a more detailed investigation into the importance of the symmetry mechanism for layout. One way this could be performed is by augmenting the aforementioned expressive @gnn:pl with the symmetry mechanism, removing ambiguity around whether expressivity is acting as a bottleneck.

Mamba showed positive results for extracting useful features from raw nucleotide data, however, this project used aggressive graph subsampling due to compute and memory limitations. Scaling-up this technique with more compute, and also using Transformer @transformer-paper as an alternative sequence-to-sequence model (as heavily-optimized versions become available), is yet another direction.

Lastly, building on the promising results from the end-to-end neural genome assembly experiments, future work could assess the method's tolerance and robustness to sequencing errors in the reads. Moreover, the Transformer-based implementation may be augmented with tokenization schemes such as DNABERT(2) @dnabert @dnabert-2, and entropy-based tokenization schemes such as the Byte Latent Transformer @byte-latent-transformer. Additionally, ultra-long reads can be provided as an additional input to resolve ambiguities in particularly complex genomic regions, such as those exhibiting tandem repeats.

// Exploring SymGAT's failure

// https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html instead of Mamba?

// Trying out more expressive architectures, to firmly know the theoretical bounds of the problem---are high expressiveness GNNs necessary.

// If we cannot computationally find a more expressive model, we can fall back to more data + regularization -> starting to become common practice in chemical field + AlphaFold 3.

<end-main-body>

#pagebreak()

#show heading.where(level: 1): it => [
  #set text(weight: "regular", size: 12pt)
  #v(8em)
  #linebreak()
  #text(size: 2em)[#it.body.text]
  #v(2em)
]

#bibliography("bibligraphy.bib")

#pagebreak()

#set heading(numbering: "A1.", supplement: "Appendix")
#counter(heading).update(0)

#show heading.where(level: 1): it => [
  #set text(weight: "regular", size: 12pt)
  #v(8em)
  #it.supplement #context counter(heading).display("A")
  #linebreak()
  #text(size: 2em)[#it.body.text]
  #v(2em)
]

#[
= Experiment setup
== Experiment hyperparameters
#let fmt_value = value => text(fill: rgb("B60158"), font: "DejaVu Sans Mono", size: 0.9em)[#value]

#[
  #set table(
    fill: (x, y) => if calc.rem(y, 2) == 1 {
        gray.lighten(65%)
      }
  )
  #table(
    columns: (0.5fr, 1fr),
    align: left,
    table.header([Hyperparameter], [Value]),
    [Optimizer],
    [```yaml 
    Adam
    Learning Rate: 0.0001```],
    [Learning rate scheduler],
    [```yaml 
    Reduce Learning Rate on Plateau
    Factor: 0.9
    Patience: 5 # epochs```],
    [Batch Size],
    [```yaml
    Single graph per batch
    Num. graphs: 1
    ```],
    [Number of epochs],
    [```yaml
    Ensure training till convergence
    Epochs: 200
    ```],
    [Loss function],
    [```yaml 
    Binary Cross Entropy with Logits + Symmetry loss
    Alpha: 0.1 # controls weighting between the two loss functions```
    ],
    [Training graph masking],
    [```yaml
    Between 0 and 10% of the nodes in the input graph (before partitioning) may be removed to simulate varying read coverage
    Mask fraction Low: 0.9
    Mask fraction High: 1.0
    ```],
    [Number of nodes per cluster],
    [```yaml
    Controls the size of partitioned subgraphs
    Mamba experiments: 600
    All other experiments: 2000
    ```],
    [Seed],
    [```yaml
    All experiments were repeated 5 times
    Seeds: [0, 1, 2, 3, 4]
    ```]
  )
]

== Dataset generation
Data for all chromosomes was sourced from the `CHM13v2` @t2t human genome assembly (`BioProject PRJNA559484` @chm13-acrocentric) from the #underline[#link("https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_009914755.1/", [National Library of Medicine])]. Although the human genome is diploid, this assembly is haploid, as it was derived from a hydatidiform mole.

Both @pacbio @hifi and @ont @ul reads were simulated from the above reference genome using `PBSIM3` @pbsim3, with `Hifiasm` @hifiasm-paper used for the assembly of all overlap graphs (including those integrating ultra-long reads @double-graph). The data generation pipeline used a modified version of GNNome @lovrolong.
#pagebreak()

= Additional algorithms <app:algorithms>
#[
  #set table.cell(align: left)
  #set text(size: 12.5pt)
  #algorithm_2
]

= Additional experimental data
== Integration of ultra-long data and GRANOLA <app:ultra-long-granola>
The tables below contain the experimental results of running the @gatedge and @symgatedge architectures on overlap graphs with ultra-long data, and with @granola additionally added.

=== Chromosome 19 testing dataset <app:chr-19-ul-granola-test>
#figure(
table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header([Metric (%)], [@gatedge], [@gatedge (@ul)], [@gatedge (@ul+@granola)]),
  [$arrow.t$ Accuracy], best[#a_sd(0.852172431936537, 0.018122915137703973)], [#a_sd(0.8400341272347521, 0.004585857523053335)], [#a_sd(0.8263452834056321, 0.005971217304973857)],
  [$arrow.t$ Precision], best[#a_sd(0.9491644376556241, 0.9491644376556241)], [#a_sd(0.910062909230493, 0.008397492689877816)], [#a_sd(0.9202283189065109, 0.004259045323413266)],
  [$arrow.t$ Recall], [#a_sd(0.8816506171113823, 0.021949458751530884)], best[#a_sd(0.8936830314971256, 0.01564208464072007)], [#a_sd(0.8634125934522685, 0.012421249272288383)],
  [$arrow.t$ F1], best[#a_sd(0.914044335828298, 0.011599446124280127)], [#a_sd(0.9016702165714898, 0.004092508778724496)], [#a_sd(0.8908520470838276, 0.0047318961491994195)],
  highlight[$arrow.t$ Precision Inverse], highlight[#a_sd(0.3848171812904071, 0.038358039650424273)], highlight(best[#a_sd(0.5503454799288457, 0.015864438246476206)]), highlight[#a_sd(0.5121686449913767, 0.013812536922495972)],
  [$arrow.t$ Recall Inverse], [#a_sd(0.6066977066096203, 0.015938260139693062)], [#a_sd(0.5938508220577763, 0.04789138547267719)], best[#a_sd(0.6562513546149797, 0.02459306289512086)],
  highlight[$arrow.t$ F1 Inverse], highlight[#a_sd(0.46957065155016847, 0.02481152760711465)], highlight[#a_sd(0.5699557089584116, 0.013585702239012403)], highlight(best[#a_sd(0.5748741374233737, 0.0032808270517395224)]),
),
caption:[There is a significant performance uplift when ultra-long data is integrated into the overlap graph (@gatedge), as was observed with @symgatedgcn, with much higher inverse precision and F1 score #highlighted. @granola does not help in improving performance further. The results show the mean and standard deviation across 5 runs, with chromosome 15 used for training, and 9 as the testing dataset for these metrics. Best results in *bold*. $arrow.t$ indicates _higher is better_.]
) <tab:gat_test>

#figure(
table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header([Metric (%)], [@symgatedge], [@symgatedge (@ul)], [@symgatedge (@ul+@granola)]),
  [$arrow.t$ Accuracy], best[#a_sd(0.8420853457812795, 0.017519697090493506)], [#a_sd(0.8364363632334447, 0.003329035545859219)], [#a_sd(0.8146057362563088, 0.00581974743084867)],
  [$arrow.t$ Precision], best[#a_sd(0.9526119974753542, 0.001415315356238136)], [#a_sd(0.9203417460861938, 0.002873804276141712)], [#a_sd(0.9225942806952138, 0.0033192279524718615)],
  [$arrow.t$ Recall], [#a_sd(0.8662298215722646, 0.02142411761762067)], best[#a_sd(0.8766909125259776, 0.0060060677440282005)], [#a_sd(0.8451556638147418, 0.011116819171108488)],
  [$arrow.t$ F1], best[#a_sd(0.9072501961743051, 0.011465972986112126)], [#a_sd(0.8979707758893195, 0.002408303185493175)], [#a_sd(0.882129776316059, 0.004652621452553112)],
  highlight[$arrow.t$ Precision Inverse], highlight[#a_sd(0.36794115048395104, 0.030808877124403163)], highlight(best[#a_sd(0.535446470191115, 0.008795377752307989)]), highlight[#a_sd(0.48738144691836205, 0.01073288066497148)],
  [$arrow.t$ Recall Inverse], [#a_sd(0.641026205681568, 0.017880694079863063)], [#a_sd(0.6517168777285816, 0.015402737588688454)], best[#a_sd(0.6744186766572746, 0.019091024109722618)],
  highlight[$arrow.t$ F1 Inverse], highlight[#a_sd(0.4665307562943479, 0.02181144060394172)], highlight(best[#a_sd(0.5877544298868103, 0.006282638788092146)]), highlight[#a_sd(0.5655748649808248, 0.002447230862814821)],
),
caption: [Likewise for the @symgatedge architecture.]
) <tab:symgat_test>

#pagebreak()

=== Chromosome 19 assembly compared with reference  <app:chr-19-ul-granola-assembly>

#figure(
table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header([Assembly Metric], [@gatedge], [@gatedge (@ul)], [@gatedge (@ul+@granola)]),
  [$arrow.b$ Num. contigs], best[#a_sd_a(base-chr9-GAT-contigs)], [#a_sd_a(ul-chr9-GAT-contigs)], [#a_sd_a(granola-ul-chr9-GAT-contigs)],
  [$arrow.t$ Longest contig length], best[#a_sd_a(base-chr9-GAT-largest-contig, multiplier: 0.0000001)], [#a_sd_a(ul-chr9-GAT-largest-contig, multiplier: 0.0000001)], [#a_sd_a(granola-ul-chr9-GAT-largest-contig, multiplier: 0.0000001)],
  highlight[$arrow.t$ Genome fraction (%)], highlight[#a_sd_a(base-chr9-GAT-genome-fraction)], highlight[#a_sd_a(ul-chr9-GAT-genome-fraction)], highlight(best[#a_sd_a(granola-ul-chr9-GAT-genome-fraction)]),
  [$arrow.t$ NG50], best[#a_sd_a(base-chr9-GAT-ng50, multiplier: 0.0000001)], [#a_sd_a(ul-chr9-GAT-ng50, multiplier: 0.0000001)], [#a_sd_a(granola-ul-chr9-GAT-ng50, multiplier: 0.0000001)],
  // [NGA50], best[#a_sd_a(base-chr9-GAT-nga50, multiplier: 0.0000001)], [#a_sd_a(ul-chr9-GAT-nga50, multiplier: 0.0000001)], [#a_sd_a(granola-ul-chr9-GAT-nga50, multiplier: 0.0000001)],
  [$arrow.b$ Num. misma. (per 100 @kb)], best[#a_sd_a(base-chr9-GAT-mismatches)], [#a_sd_a(ul-chr9-GAT-mismatches)], [#a_sd_a(granola-ul-chr9-GAT-mismatches)],
  [$arrow.b$ Num. indels (per 100 @kb)], [#a_sd_a(base-chr9-GAT-indels)], [#a_sd_a(ul-chr9-GAT-indels)], best[#a_sd_a(granola-ul-chr9-GAT-indels)],
),
caption: [Integration of ultra-long data into the overlap graph (@gatedge (@ul)) results in a higher fraction of the reference genome being covered in the reconstructed assembly #highlighted, as was observed with @symgatedgcn. This is achieved whilst maintain assembly quality. @granola is detrimental to assembly quality. The results show the mean and standard deviation across 5 runs, with chromosome 15 used for training, and 9 as the reference genome assembled. Best results in *bold*. $arrow.t$ indicates _higher is better_. $arrow.b$ indicates _lower is better_.]
) <tab:gat_assembly>



#figure(
table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header([Assembly Metric], [@symgatedge], [@symgatedge (@ul)], [@symgatedge (@ul+@granola)]),
  [$arrow.b$ Num. contigs], [#a_sd_a(base-chr9-SymGAT-contigs)], best[#a_sd_a(ul-chr9-SymGAT-contigs)], [#a_sd_a(granola-ul-chr9-SymGAT-contigs)],
  [$arrow.t$ Longest contig length], [#a_sd_a(base-chr9-SymGAT-largest-contig, multiplier: 0.0000001)], best[#a_sd_a(ul-chr9-SymGAT-largest-contig, multiplier: 0.0000001)], [#a_sd_a(granola-ul-chr9-SymGAT-largest-contig, multiplier: 0.0000001)],
  highlight[$arrow.t$ Genome fraction (%)], highlight[#a_sd_a(base-chr9-SymGAT-genome-fraction)], highlight(best[#a_sd_a(ul-chr9-SymGAT-genome-fraction)]), highlight[#a_sd_a(granola-ul-chr9-SymGAT-genome-fraction)],
  [$arrow.t$ NG50], [#a_sd_a(base-chr9-SymGAT-ng50, multiplier: 0.0000001)], best[#a_sd_a(ul-chr9-SymGAT-ng50, multiplier: 0.0000001)], [#a_sd_a(granola-ul-chr9-SymGAT-ng50, multiplier: 0.0000001)],
  // [NGA50], best[#a_sd_a(base-chr9-SymGAT-nga50, multiplier: 0.0000001)], [#a_sd_a(ul-chr9-SymGAT-nga50, multiplier: 0.0000001)], [#a_sd_a(granola-ul-chr9-SymGAT-nga50, multiplier: 0.0000001)],
  [$arrow.b$ Num. misma. (per 100 @kb)], [#a_sd_a(base-chr9-SymGAT-mismatches)], best[#a_sd_a(ul-chr9-SymGAT-mismatches)], [#a_sd_a(granola-ul-chr9-SymGAT-mismatches)],
  [$arrow.b$ Num. indels (per 100 @kb)], [#a_sd_a(base-chr9-SymGAT-indels)], [#a_sd_a(ul-chr9-SymGAT-indels)], best[#a_sd_a(granola-ul-chr9-SymGAT-indels)],
),
caption: [Likewise for the @symgatedge architecture.]
) <tab:symgat_assembly>
] <no-wc>