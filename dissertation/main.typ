#import "@preview/glossarium:0.5.6": make-glossary, register-glossary, print-glossary, gls, glspl
#import "@preview/wordometer:0.1.4": word-count, total-words

#set page(margin: ("top": 20mm, "bottom": 20mm, "left": 25mm, "right": 25mm))
#set text(size: 12pt)
#set heading(numbering: "1")

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
#set page(numbering: "i")

#title[Declaration]

I, Yash Shah of Gonville & Caius College, being a candidate for Part III of the Computer Science Tripos, hereby declare that this report and the work described in it are my own work, unaided except as may be specified below, and that the report does not contain material that has already been used to any substantial extent for a comparable purpose. In preparation of this report, I adhered to the Department of Computer Science and Technology AI Policy. I am content for my report to be made available to the students and staff of the University.

Signed [signature]

Date [date]

#total-words

#pagebreak()

#title[Abstract]

#pagebreak()

#title[Acknowledgements]

#pagebreak()

#set heading(supplement: [Chapter])

#show heading.where(level: 1): it => [
  #set text(weight: "regular", size: 12pt)
  #v(8em)
  #text(size: 2em)[#it.body.text]
  #v(2em)
]

#outline()

#pagebreak()

#title[Acronyms]

#show: make-glossary
#import "glossary.typ": entry-list
#register-glossary(entry-list)
#print-glossary(
 entry-list
)

#show heading.where(level: 1): it => [
  #set text(weight: "regular", size: 12pt)
  #v(8em)
  #it.supplement #context counter(heading).display("1")
  #linebreak()
  #text(size: 2em)[#it.body.text]
  #v(2em)
]

#counter(page).update(1)
#set page(numbering: "1")
#set math.equation(numbering: "(1)")
#show: word-count

#set heading(numbering: "1.")

= Introduction
== Motivation
Genome assembly has remained a central subject in computational biology for the past four decades, as accurate reconstruction of an organism’s genome is essential in understanding its biology and evolution. By enabling researchers to map and analyze an organism's @dna, including genes, regulatory elements, and non-coding regions (segments that do not directly encode proteins), we gain insight into the organism's traits, development, and overall function. Comparative analyses of genome assemblies across species also sheds light on evolutionary relationships.

In addition, assembling a large number of genomes from the same species allows scientists to study the role of genetic variation in health and disease, revealing factors that contribute to susceptibility or resistance to various conditions. This is increasingly important as we move into the realm of targeted healthcare, such as personalized drugs that are tailored to an individual by utilizing their unique genetic information, providing more effective treatment. Genome assemblies are often crucial pre-requisites for downstream biological analysis, and in this project, we demonstrate how machine learning is an effective tool in improving de novo (without relying on a pre-existing reference genome) assembly, by increasing accuracy, scalability, and speed, as well as reducing costs.

== Existing methods <sec:existing_methods>
Although the Human Genome Project concluded in 2003, the first @t2t sequencing of human DNA was only achieved in 2021. A @t2t sequence is a gapless, unfragmented genome assembly, necessary for understanding the full structural complexity of a chromosome, with telomeres being the protective structures found at the ends of chromosomes. 

Historically, hierarchical sequencing and @wgs have been the two predominant strategies. Hierarchical sequencing involves cloning, sequencing, and assembly of tiled genomic fragments that are aligned against a physical or genetic genome map, with the human reference genome GRCh38 being primarily constructed with this method. 

Due to its high cost and labour-intensive nature, heirarchical sequencing has largely been replaced by @wgs, where the genome is randomly fragmented into individually sequenced smaller segments called reads. These reads are then reassembled into a complete genome by identifying overlaps between them. Unlike hierarchical sequencing, @wgs must consider overlaps between reads spanning the entire genome, not just localized regions, which significantly increases computational complexity.

A fundamental part of @wgs is the creation of the overlap graph, in which each vertex is a read. There exists a directed edge between the vertex of read $A$ and of read $B$ if the suffix of $A$ can be aligned to (i.e. overlaps with) the prefix $B$. However, in practice, this overlap graph is not perfect. Due to the computational cost of exact overlap calculations and the inherent noise in sequencing technologies, overlaps are imprecise. Additional challenges include errors in base-calling (translating the electrical signals into a sequence of nucleotides: @nuc_a, @nuc_g, @nuc_c, @nuc_t) the raw read data, long repetitive regions in the genome, and other sequencing artifacts---all of which introduce spurious nodes and edges into the graph, which must be cleaned up.

Existing methods for overlap graph simplification involve a collection of algorithms and heuristics to remove artifacts such as bubbles, dead-ends, and transitive edges. Despite their utility, these methods often struggle in complex genomic regions, where unique assembly solutions may not exist, resulting in either the omission of these complex regions in the final assembly completely, fragmenting the resulting genome assembly, or reliance on manual curation by human experts---an approach that is time-consuming, costly, and not scalable when processing thousands of genomes.

== Related work

== Aims

== Key contributions


Motivation
  - Applications
    - Could talk about in the future being able to integrate other types of read data
  
  - Existing methods
    - Little work done on using neural networks
    - Talk about the heuristics that are used possibly

Outline
  - Aims
  - Key contributions

#pagebreak()

= Background
== Read technology
There are three key characteristics of sequencing reads that are often traded-off during genome assembly: length, accuracy, and evenness of representation. Contemporary efforts targeting de novo @t2t assembly focus on accurate long-read technology that produces contiguous sequences spanning $>=10$ @kb in length, with @pacbio and @ont being the two companies leading their development. This project utilizes @pacbio's @hifi read technology, due to its potential to generate long reads spanning $10$--$20$ @kb in length, whilst maintaining a low error rate of $<0.5%$, and is the current core data type for high-quality genome assembly.

== Overlap-Layout-Consensus //https://bio.libretexts.org/Bookshelves/Computational_Biology/Book%3A_Computational_Biology_-_Genomes_Networks_and_Evolution_(Kellis_et_al.)/05%3A_Genome_Assembly_and_Whole-Genome_Alignment/5.02%3A_Genome_Assembly_I-_Overlap-Layout-Consensus_Approach
The fundamental problem in genome sequencing is that no current technology that can read continuously from one end of the genome to the other. Instead, sequencing technologies only produce relatively short contiguous fragments called reads. Most chromosomes are $>10$ @mb long, and can be up to $1$ @gb long, while even current long-read sequencing technologies only produce accurate reads up to a few $10$s of @kb. Thus assembling the genome requires an algorithm to combine these shorter reads. @olc is the predominant approach for genome assembly with long reads. In this section, we discuss the three phases of @olc in more detail.

=== Overlap
The first step is identifying overlapping reads. Read $A$ overlaps with read $B$ if the suffix of $A$ matches the prefix of $B$. While the Needleman-Wunsch dynamic programming algorithm can be used to find overlaps through pairwise alignments of reads, its $cal(O)(n^2)$ (where $n$ is the nucleotide length of the longer read) complexity for each pair of reads makes it infeasible for genome assembly involving millions, or billions of read pairs. Moreover, most read pairs do not overlap, making exhaustive pairwise alignment highly inefficient.

#let kmer = [$k$-mer]
#let kmers = [$k$-mers]
Instead, the BLAST algorithm is used, leveraging #kmers --- unique $k$-length substrings acting as seeds for identifying overlaps. The algorithm extracts all #kmers from the reads and locates positions where multiple reads share common #kmers. An approximate similarity score is computed depending on the multiplicity and location of matching #kmers.

Next, read pairs (matches) falling under some threshold of similarity, say $95%$, are discarded. The full alignment need only be calculated for these remaining matching reads. The matches do not need to be identical, allowing tolerance for sequencing errors (and heterozygosity for diploid(/polyploid) organisms (like humans) where there may be two variants of an allele with one from each parent at polymorphic sites in the genome).

This overlap information is used to construct the overlap graph in which each vertex is a read and there exists a directed edge between the vertex of read $A$ and of read $B$ if they overlap.

=== Layout
In a perfect overlap graph, free from artifacts, the genome can be reconstructed by finding a Hamiltonian path (a path that visits every vertex/read in the graph exactly once). Contemporary assemblers first simplify the overlap graph by removing spurious vertices and edges (such as bubbles, dead-ends, and transitive edges), aiming to simplify the graph into a chain. However, as previously mentioned in @sec:existing_methods, this simplification is often incomplete, or infeasible.

This project targets this layout phase with the use of @gnn:pl. The @gnn takes a partially simplified overlap graph as input, and predicts a probability of each edge belonging to the Hamiltonian path corresponding to the genome.

=== Consensus
In this final phase, the remaining reads are mapped onto a linear assembly and per-base errors are addressed, for example by taking the most common base substring for a region where several reads overlap.

== Geometric Deep Learning
@gdl is a framework leveraging the geometry in data, through groups, representations, and principles of invariance and equivariance, to learn more effective machine learning models. 

Central to @gdl are symmetries---transformations that that leave an object unchanged. In the context of machine learning, relevant symmetries can arise in various forms: symmetries of the input data (e.g. rotational symmetries in molecular structure); the label function mapping the input to some output (e.g. the image classification function is invariant to the location of the object in the image), the domain our data lives on (e.g. data living on a set is invariant to the permutation of items in the set), or even symmetries in the model's parameterization.

The key insight is that by encoding symmetry within our model architecture, we restrict the space of functions that can be represented to those that respect these symmetries. This makes models more performant, improves generalization, and can make learning more sample/data efficient.

Within genome assembly, we operate on input overlap graphs. By studying the symmetries of graphs by inspecting their invariances and equivariances, we develop a machine learning architecture, the @gnn, that is tailored to operate effectively on graph-structured data.
=== Permutation Invariance and Equivariance
Let $G = (V, E)$ be a graph such that $V$ is the set of nodes representing arbitrary entities. $E subset.eq V times V$ is the set of edges such that $(u, v) in E$ encodes relationships among these nodes/entities. The complete connectivity of $G$ has an algrebric representation $bold(A) in RR^(|V| times |V|)$, the adjacency matrix such that:
$ A_(u v) = cases(1", if"  space (v, u) in cal(E),
                  0", if" space (v, u) in.not E) $

Now assume that each node $v$ is equipped with a feature vector $bold(x)_v$. By stacking these per-node feature vectors, we get the node feature matrix $bold(X) = (bold(x_1), ..., bold(x_n))^T in |V| times k$, where $k$ is the feature dimension, with the $v^"th"$ row corresponding to $bold(x)_v$.

#let perm = $bold(P)$
#let features = $bold(X)$
#let adj = $bold(A)$

Given an arbitrary permutation matrix #perm, a function $f$ is said to be permutation _invariant_ iff
$f(perm features, perm adj perm^T) = f(features, adj)$. Likewise, a function $bold(F)$ is said to be permutation _equivariant_ iff $bold(F)(perm features, perm adj perm^T) =  perm bold(F)(features, adj)$.

=== Graph Neural Networks
#let neighborhood = $cal(N)$
Let $G = (V, E)$ be a graph, with $neighborhood_v = {u in V : (v,u) in E}$ representing the one-hop neighborhood  of node $v$, having node features $features_(neighborhood_v) = {{bold(x)_u: u in neighborhood_v}}$, where ${{dot}}$ denotes a multiset. 
We define $f$, the message passing function, as a local and permutation-invariant function over the neighborhood features $features_(neighborhood_u)$ as:
$ f(bold(x)_v, features_(neighborhood_v)) = phi.alt(bold(x)_v, plus.circle.big_(u in neighborhood_v) psi(bold(x)_v, bold(x)_u)) $

where $psi$ and $phi.alt$ are learnable message, and update functions, respectively, while $plus.circle$ is a permutation-invariant aggregation function (e.g., sum, mean, max). A permutation-equivariant GNN layer $bold(F)$ is the local message passing function applied over all neighborhoods of G:
$ bold(F)(features, adj) = mat(dash.em f(bold(x)_1, features_(neighborhood_1)) dash.em;
                               dash.em f(bold(x)_2, features_(neighborhood_2)) dash.em;
                               dots.v;
                               dash.em f(bold(x)_n, features_(neighborhood_n)) dash.em;) $

A @gnn consists of sequentially applied message passing layers.

== Mamba Selective State Space Model
Mamba is derived from the class of Structured State Space Models (S4) @mamba, combining aspects of recurrent and convolutional networks. While S4 models have a recurrent formulation, a parallelizable convolutional operation applied to a sequence yields the identical result, making them much faster than previous @rnn architectures, such as @lstm networks.

Crucially, S4 models scale better with sequence length in comparison to other parallel architectures such as Transformer. While Transformers incur quadratic complexity with sequence length, S4 models have linear, or near-linear scaling. Moreover, these models have principled mechanisms for long-range dependency modelling @ssm-long-range, critical for sequence modelling, and perform well in benchmarks such as Long Range Arena @long-range-arena. 

Formally, S4 models, defined with continuous-time parameters $(Delta, bold(A), bold(B), bold(C)))$ can be formulated as follows:
$ h'(t) = bold(A) h(t) + bold(B) x(t) $
$ y(t) = bold(C) h(t) $

These equations refer to a continuous-time system, mapping a _continuous _ sequence $x(t) in bb(R) arrow.r y(t) in bb(R)$, through an implicit hidden latent space $h(t) in bb(R)^N$. For discrete data, like a sequence of bases in a read, these equations are discretized using the step size $Delta$, transforming the continuous-time parameters $(Delta, bold(A), bold(B))$ into discrete-time parameters $(bold(dash(A)), bold(dash(B)))$ through a discretization rule. This yields a new set of discrete equations:
$ h_t = dash(bold(A))h_(t - 1) + dash(bold(B)) x_t $
$ y_t = bold(C)h_t $

Through repeated application of the recurrence relation, and simplification via the @lit property (which states that $(Delta, bold(A), bold(B))$ and consequently $(bold(dash(A)), bold(dash(B)))$ remain constant for all time-steps), the system can be equivalently expressed as a 1-dimensional convolution over the sequence $x$ with kernel $bold(dash(K))$ ($star$ denotes the covolution operation):
$ bold(dash(K)) = (C dash(B), C dash(A) dash(B), ..., C dash(A)^k dash(B), ...) $
$ y = x star bold(dash(K)) $



Although S4 can model recurrent processes with latent state across large contexts well, they are not as effective at modelling discrete, information-dense data. To address this, Mamba extends the S4 formulation by incorporating _selectivity_---the ability to select data in an input-dependent manner, helping filter out irrelevant data, and keep relevant information indefinitely. However, this breaks the time- and input-invariance that makes the convolution operation possible in S4, responsible for S4's speed. This is compensated for by replacing the convolution with a scan/prefix sum operation. 

The ability to select data in an input-dependent manner, along with the scan/prefix sum in-place of convolution, together results in the Mamba _Selective_ State Space Model (henceforth referred to simply as Mamba).






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
Brief overview of the entire process

Detailed explanation of the dataset generation, which involves simuation of the reads, preprocessing the real reads, constructing the graph

Also discussion of the input features provided to the GNN
  - Emphasis on the use of the actual read data

Discussion of the GNN architecture
  - Overview of SymGatedGCN---particularly the importance of the Sym part
  - Overview of the new GAT, and SymGAT architectures, that also use the edge features (the original GAT model does not use)
  - Overview of GRANOLA---how it uses MPNN + RNF was maximizing expressiveness

Discussion of Mamba SSM
  - Why it was chosen over say using attention
  - Problems with tokenization of DNA

Explanation of the decoding process to reconstruct the genome



#pagebreak()

= Evaluation

#pagebreak()

= Summary and conclusions

#pagebreak()

#bibliography("bibligraphy.bib")

#pagebreak()

#set heading(numbering: "A", supplement: "Appendix")
#counter(heading).update(0)

#show heading.where(level: 1): it => [
  #set text(weight: "regular", size: 12pt)
  #v(8em)
  #it.supplement #context counter(heading).display("A")
  #linebreak()
  #text(size: 2em)[#it.body.text]
  #v(2em)
]

= Experiment setup