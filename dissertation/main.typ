#import "@preview/glossarium:0.5.6": make-glossary, register-glossary, print-glossary, gls, glspl
#import "@preview/wordometer:0.1.4": word-count, total-words
#import "@preview/algorithmic:0.1.0"
#import algorithmic: algorithm
#import "@preview/subpar:0.2.2"

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
#set page(numbering: "i", background: none)

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
#set place(float: true)
#show figure.caption: set text(size: 0.9em)
#show figure.caption: set align(left)
#let subcaption = it => [
  #set align(center)
  #v(-1.9em)
  #it
  #v(1em)
]
#let subcaptionlong = it => [
  #set align(center)
  #v(-1.85em)
  #h(1.5em)
  #it
  #v(1em)
]

= Introduction
== Motivation
Genome assembly has remained a central subject in computational biology for the past four decades, as accurate reconstruction of an organism’s genome is essential in understanding its biology and evolution. By enabling researchers to map and analyze an organism's @dna, including genes, regulatory elements, and non-coding regions (segments that do not directly encode proteins), we gain insight into the organism's traits, development, and overall function. Comparative analyses of genome assemblies across species also sheds light on evolutionary relationships.

In addition, assembling a large number of genomes from the same species allows scientists to study the role of genetic variation in health and disease, revealing factors that contribute to susceptibility or resistance to various conditions. This is increasingly important as we move into the realm of targeted healthcare, such as personalized drugs that are tailored to an individual by utilizing their unique genetic information, providing more effective treatment. 

A @t2t assembly is essential in gaining complete insight into an organism's genome. A @t2t sequence represents a complete, continuous genome without gaps or fragmentation. Remarkably, achieving such assemblies has only become feasible in recent years with the advent of long-read sequencing technologies. Although the Human Genome Project concluded in 2003, the reference genome produced contained several missing regions that were challenging to assemble, with the first @t2t sequencing of human DNA only recently achieved in 2021.

@T2t assemblies are a foundational requirement for several downstream biological analysis, and in this project, we demonstrate how machine learning is an effective tool in improving de novo (without relying on a pre-existing reference genome) @t2t assembly, by increasing accuracy, scalability, and speed, as well as reducing costs.

== Existing methods <sec:existing_methods>
Historically, hierarchical sequencing and @wgs have been the two predominant assembly strategies. Hierarchical sequencing involves cloning, sequencing, and assembly of tiled genomic fragments that are aligned against a physical or genetic genome map, with the human reference genome GRCh38 being primarily constructed with this method. 

Due to its high cost and labour-intensive nature, heirarchical sequencing has largely been replaced by @wgs, where the genome is randomly fragmented into individually sequenced smaller segments called reads. These reads are then reassembled into a complete genome by identifying overlaps between them. Unlike hierarchical sequencing, @wgs must consider overlaps between reads spanning the entire genome, not just localized regions, which significantly increases computational complexity.

A fundamental part of @wgs is the creation of the overlap graph, in which each vertex is a read. There exists a directed edge between the vertex of read $A$ and of read $B$ if the suffix of $A$ can be aligned to (i.e. overlaps with) the prefix $B$. However, in practice, this overlap graph is not perfect. Due to the computational cost of exact overlap calculations and the inherent noise in sequencing technologies, overlaps are imprecise.

Additional challenges include errors in base-calling (translating the electrical signals into a sequence of nucleotides: @nuc_a, @nuc_g, @nuc_c, @nuc_t) the raw read data, long repetitive regions in the genome, and other sequencing artifacts---all of which introduce spurious nodes and edges into the graph, which must be cleaned up.

#place(top + center)[#figure(image("graphics/artifacts.svg", width: 90%), caption: [Elementary artifact types encountered in overlap graphs. Each node in this example overlap graph represents a read, and the edges correspond to overlaps between those reads.]) <fig:common_artifacts>]

Existing methods for overlap graph simplification involve a collection of algorithms and heuristics to remove elementary artifacts such as bubbles, dead-ends, and transitive (shortcut) edges (@fig:common_artifacts). It is vital to note that artifacts occurring in overlap graphs are not bound to only these three categories, and do not occur in isolation. Instead, they frequently have complex interactions that lead to challenging to resolve tangles (demonstrated in @tip and @tangle). 

Consequently, despite the utility of heuristic algorithms, these methods often struggle in complex genomic regions, where unique assembly solutions may not exist, resulting in either the omission of these complex regions in the final assembly completely, leading to a fragmented and incomplete result, or reliance on manual curation by human experts---an approach that is time-consuming, costly, and not scalable when processing thousands of genomes.

#place(top + center)[#subpar.grid(
  columns: (1fr, 1fr, 1fr),
  figure(image("graphics/chr19_bubble.png"), caption: subcaption[
    Bubble
  ]), <bubble>,
  figure(image("graphics/chr21_tip.png"), caption: subcaption[
    Dead-end/Tip
  ]), <tip>,
  figure(image("graphics/chr19_tangle.png"), caption: subcaption[
    Transitive edges
  ]), <tangle>,
  caption: [The figures show common artifacts in overlap graphs generated from real read data, taken from human chromosome 19 (a, b) and 21 (c). Note that these are the very same overlap graphs that are successfully simplified and resolved by our work.],
  label: <full>,
)]

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


== Overlap-Layout-Consensus //https://bio.libretexts.org/Bookshelves/Computational_Biology/Book%3A_Computational_Biology_-_Genomes_Networks_and_Evolution_(Kellis_et_al.)/05%3A_Genome_Assembly_and_Whole-Genome_Alignment/5.02%3A_Genome_Assembly_I-_Overlap-Layout-Consensus_Approach
The fundamental problem in genome sequencing is that no current technology that can read continuously from one end of the genome to the other. Instead, sequencing technologies only produce relatively short contiguous fragments called reads. Most chromosomes are $>10$ @mb long, and can be up to $1$ @gb long, while even current long-read sequencing technologies only produce accurate reads up to a few $10$s of @kb. Thus assembling the genome requires an algorithm to combine these shorter reads. @olc is the predominant approach for genome assembly with long reads. In this section, we discuss the three phases of @olc in more detail.

#let kmer = [$k$-mer]
#let kmers = [$k$-mers]

#place(bottom + center)[#subpar.grid(
    columns: 2,
    gutter: 4em,
    figure(image("graphics/overlap.svg"), caption: subcaption[Overlap between two reads]), <fig:overlap>,
    figure(image("graphics/kmer.svg"), caption: subcaptionlong[All 3-mers present in a read]), <fig:kmer>,
    caption: [(a) shows the maximal overlapping region between a pair of reads, where the alignment is found using the Needleman-Wunsch dynamic programming algorithm. (b) shows all #kmers present in an example read, where $k=3$.]
)]


=== Overlap
The first step is identifying overlapping reads. Read $A$ overlaps with read $B$ if the suffix of $A$ matches the prefix of $B$ (shown in @fig:overlap). While the Needleman-Wunsch dynamic programming algorithm can be used to find overlaps through pairwise alignments of reads, its $cal(O)(n^2)$ (where $n$ is the nucleotide length of the longer read) complexity for each pair of reads makes it infeasible for genome assembly involving millions, or billions of read pairs. Moreover, most read pairs do not overlap, making exhaustive pairwise alignment highly inefficient.


Instead, the BLAST algorithm is used, leveraging $k$-mers---unique $k$-length substrings (example #kmers for a read are displayed in @fig:kmer) acting as seeds for identifying overlaps. The algorithm extracts all #kmers from the reads and locates positions where multiple reads share common #kmers. An approximate similarity score is computed depending on the multiplicity and location of matching #kmers.

Next, read pairs (matches) falling under some threshold of similarity, say $95%$, are discarded. The full alignment need only be calculated for these remaining matching reads. The matches do not need to be identical, allowing tolerance for sequencing errors (and heterozygosity for diploid(/polyploid) organisms (like humans) where there may be two variants of an allele with one from each parent at polymorphic sites in the genome).

This overlap information is used to construct the overlap graph in which each vertex is a read and there exists a directed edge between the vertex of read $A$ and of read $B$ if they overlap (@fig:layout_phase).

=== Layout
In a perfect overlap graph, free from artifacts, the genome can be reconstructed by finding a Hamiltonian path (a path that visits every vertex/read in the graph exactly once). Contemporary assemblers first simplify the overlap graph by removing spurious vertices and edges (such as bubbles, dead-ends, and transitive edges), aiming to simplify the graph into a chain. @fig:sequencing_errors shows how some of these errors form. However, as previously mentioned in @sec:existing_methods, this simplification is often incomplete, or infeasible.

#place(top + center)[#subpar.grid(
  columns: (1fr, 0.9fr, 1fr),
  align: top,
  gutter: 2em,
  figure(image("graphics/layout_phase.svg", height: 8cm, fit: "contain"), caption: subcaptionlong[Constructing the Overlap Graph]), <fig:layout_phase>,
  figure(image("graphics/sequencing_errors.svg", height: 8cm, fit: "contain"), caption: subcaptionlong[Errors manifesting in the Overlap Graph]), <fig:sequencing_errors>,
  figure(image("graphics/resolving_repeats.svg", height: 8cm, fit: "contain"), caption: subcaption[Assembling tandem #linebreak() repeat regions]), <fig:resolving_repeats>,
  caption: [All figures show the reference genome to illustrate the relative genomic positions of the reads. (a) demonstrates the construction of the overlap graph from reads. Note that there are transitive edges in the resulting graph despite the presence of sequencing errors. Unitigging refers to the process of identifying a high-confidence contig---a contiguous sequence of DNA. (b) shows how artifacts manifest in the overlap graph. A sequencing error in Read 2 leads to the formation of a tip. The #text(fill: green.darken(20%))[green] regions in the reference genome correspond to segmental duplications that cannot be distinguished. Thus, the start of Read 4 appears identical to that of Read 1, leading to the creation of an erroneous transitive edge. (c) shows how a tandem repeat region in the genome (in #text(fill: red.lighten(20%))[pink]) can be resolved by using mutations in the @dna to differentiate different positions in the repetitive region. This requires exact matches to avoid addition of erroneous edges, however this exact matching can only be performed with high accuracy reads.]
)]




This project targets this layout phase with the use of @gnn:pl. The @gnn takes a partially simplified overlap graph as input, and predicts a probability of each edge belonging to the Hamiltonian path corresponding to the genome.



=== Consensus
#place(top + center)[#figure(
  image("graphics/consensus.svg"),
  caption: [The consensus phase is responsible for the correction of per-base errors within contigs identified by the layout phase. Nucleotides in #text(fill: red)[red] highlight differences compared to the majority at that position.]
)]

Each path found in the previous layout phase corresponds to a contig---a contiguous sequence of @dna constructed from the set of overlapping reads in the path, representing a region of the genome. However, recall that the reads are erroneous, and overlaps are inexact---consensus is the step to address per-base errors.

At any particular location within the contig, there may be multiple overlapping reads. These reads need to be aligned at the base level, and then consensus on each nucleotide reached to produce the final contig. There are multiple methods of achieving consensus post read alignment. For example, a simple majority vote can be taken for each nucleotide position, or a weighted scheme, using nucleotide quality scores as weights.

== Repeating regions and the need for accurate long-read technology <sec:need_long_reads>
// https://pmc.ncbi.nlm.nih.gov/articles/PMC1226196/
A sequence occurring multiple times in the genome is known as a repetitive sequence, or repeat, for short. The repeat structure, rather than the genome's length, is the predominant determinant for the difficulty of assembly. Problematic repeating regions often consist of satellite repeats---short ($<100$ base pairs), almost identical @dna sequences that are repeated in tandem, as well as segmental duplications (also known as low-copy repeats) which are long ($1$--$400$ @kb) @dna regions that occur at multiple sites in the genome. A sufficiently long read may resolve such regions by bridging the repetitive segment and linking adjacent unique segments, however the lengths of these repetitive regions far exceed the lengths captured by any sequencing technology today. For instance, the pericentromeric region of human chromosome 1 contains 20 @mb of satellite repeats.

Fortunately, the repeat copies forming these extended repeat regions are inexact replicas due to the mutations acquired over time, and so do not share an identical repeat sequence spanning $> 10$ @kb. Although highly accurate long reads cannot span the entire region, they can distinguish between these subtly differing inexact duplicates, with their length sufficing in bridging between the differences in the repeats. With help from the @olc algorithm detailed earlier, such repeating regions can be resolved and sequenced, as demonstrated in @fig:resolving_repeats. It is critical that such long reads have high accuracy, as errors in the reads are otherwise indistinct to the mutations we rely on to sequence such regions.

Recall that @t2t assembly is a gapless reconstruction of the genome. Although long-read technology was introduced in 2010, supported by the arrival of single-molecule sequencing, their error rate ($~10%$) was too high to resolve complex genomic section, like those exhibited by repeating regions. This resulted in fragmented, and incomplete assembly. Accurate long-read technology was only available since 2019, revolutionizing genome assembly, and making possible the first @t2t human genome assembly in 2021.

== Read technology
There are three key characteristics of sequencing reads that are often traded-off during genome assembly: length, accuracy, and evenness of representation. The ideal sequencing technology produces long, highly accurate reads, with uniform coverage across the genome---avoiding gaps in low-coverage regions, and conserving computational resources in over-represented areas. Contemporary efforts targeting de novo @t2t assembly focus on accurate long-read technology that produces contiguous sequences spanning $>=10$ @kb in length, with @pacbio and @ont being the two companies leading their development.

@pacbio's @hifi read technology is the current core data type for high-quality genome assembly, due to its potential to generate reads spanning $10$--$20$ @kb in length, with an error rate $<0.5%$, replacing the previous continuous long-read solution that had an error rate $>10%$. Despite the success of long-read technology in achieving @t2t assemblies, the advent of ultra-long read technology is fast becoming a compelling additional data type to improve assembly reliability. @ont's @ul sequencing technology is central in the generation of ultra-long read data, producing reads $>100$ @kb in length, however with significantly lower accuracy than the @hifi solution.

Due to their much increased length, @ul is critical in helping resolve tangles, repeat sequences, and other artifacts that cannot be resolved with @hifi reads alone. At present, @ul reads are more expensive than @hifi data (in part as they require large amounts of input @dna), and so are not commonplace in current sequencing projects. However, as the technology matures, they offer tremendous potential in improving the accuracy and scalability of @t2t assembly. Hence, in this project, we find exploring the incorporation of such ultra-long read data with the neural genome assembly paradigm incredibly valuable.

This project utilizes @pacbio's @hifi read technology for long-read data, as well as integrating @ont @ul for ultra-long read data. The next section discusses this integration in more detail.

== Integrating ultra-long data
As shown in @sec:need_long_reads, and demonstrated in @fig:resolving_repeats, long reads are crucial in helping resolve repeating regions and tangles in assembly graphs. Longer reads are critical in improving assembly quality, but only if their accuracy is maintained. Unfortunately, current ultra-long read technology's accuracy is not high enough to replace long reads as the primary data type. Hence, we have to incorporate them as additional information into existing long-read assembly workflows. @fig:ul_strategy shows that ultra-long data can help in resolving small assembly gaps and artifacts, e.g. a bubble can be simplified by the ultra-long read bridging the bubble region, and reinforcing a unique path through that tangle.

#place(top + center)[#figure(
  image("graphics/ul-strategy.svg"),
  caption: [This figures illustrates how ultra-long read data can help improve assembly quality. (A) The #text(fill: orange)[amber] reads correspond to @pacbio @hifi long reads, and the #text(fill: purple)[purple] reads reference @ont @ul reads. Note that the ultra-longe reads contain more sequencing errors. (B) Error correction is applied to remove some sequencing errors. (C) Long-read data is used to generate an initial assembly graph, with the arrows representing sequences, with thin lines connecting those sequences. Note the presence of artifacts such as bubbles and gaps. (D) By threading ultra-long reads through this assembly graph, artifacts can be resolved, and assembly gaps patched.]
) <fig:ul_strategy>]

A naive method of ultra-long read integration is to construct two assembly graphs--- one solely from long reads, and the other from ultra-long reads. Then, these assembly graphs could be combined. Alternatively, the ultra-long reads could be treated simply as additional read data that is used to construct the assembly graph. Unfortunately, neither of these approaches would lead to a high quality assembly due to numerous issues.

Firstly, identifying correct overlaps among ultra-long reads is particularly challenging due to their higher error rate. Secondly, the @ont @ul technology in particular suffers from an increased frequency of recurrent sequence errors, making overlap identification even more problematic in complex genomic regions. Lastly, computing all-to-all pairwise overlaps is the predominant computational bottleneck in long-read assembly. Ultra-long reads increase these computational demands even further.

#place(top + center)[#figure(
  image("graphics/hifiasm_ul.svg"),
  caption: [Double graph framework used in Hifiasm (UL) to integrate @ont @ul reads with long-read information. (A) A string graph from only @pacbio @hifi reads is constructed, and ultra-long reads aligned to these long reads. (B) Ultra-long reads are translated from base-space to integer-space. (C) Overlaps between ultra-long reads are calculated in integer space, and an integer graph created. Contigs are then found in this integer graph. (D) The ultra-long contigs are integrated into the @hifi string graph. (E) Additional graph cleaning can be performed using ultra-long data. For example, the number of ultra-long reads supporting each edge can be tracked. In the case of the bubble, no ultra-long reads supported the alternative path, hence resolving the bubble. ]
) <fig:hifiasm_ul>]

An alternative approach, employed by Hifiasm (UL), which is the assembler utilized by this project, is the double graph framework (illustrated in @fig:hifiasm_ul) that exploits all information contained in both sets of reads. The @pacbio @hifi long-reads are initially used to create a string graph---an assembly graph preserving read information. Next, the @ont @ul reads are aligned to these @pacbio @hifi reads. This alignment information is then used to map the ultra-long reads from base-space into integer space---instead of each ultra-long read being a sequence of nucleotides, it is now a sequence of integer node identifiers from the @hifi string graph.

Each ultra-long read in integer space is only $10$s of node identifiers long, instead of $100$s of @kb, allowing for inexpensive all-to-all overlap calculation that is also accurate---the underlying nucleotide information is from the much more accurate @hifi reads. With ultra-long overlaps calculated, an ultra-long integer (overlap) graph can be constructed, that is then used to extract ultra-long integer contigs. These ultra-long contigs can then be incorporated into the original @hifi string graph. During this integration, the additional information provided by the ultra-long contigs can help clean the original @hifi assembly (as shown in @fig:hifiasm_ul (D)).

While the integration of ultra-long data may help eliminate some overlap graph artifacts, it introduces new erroneous nodes and edges too. This is a result of issues such as: ultra-long reads having a much higher error rate; reliance on imperfect alignment with long reads, and erroneous integer sequence overlap calculation. Ultra-long reads are poised to be a valuable data type moving forward, and so it is compelling to evaluate their utility with neural genome assembly.



== Geometric Deep Learning
@gdl is a framework leveraging the geometry in data, through groups, representations, and principles of invariance and equivariance, to learn more effective machine learning models. 

Central to @gdl are symmetries---transformations that that leave an object unchanged. In the context of machine learning, relevant symmetries can arise in various forms: symmetries of the input data (e.g. rotational symmetries in molecular structure); the label function mapping the input to some output (e.g. the image classification function is invariant to the location of the object in the image), the domain our data lives on (e.g. data living on a set is invariant to the permutation of items in the set), or even symmetries in the model's parameterization.

The key insight is that by encoding symmetry within our model architecture, we restrict the space of functions that can be represented to those that respect these symmetries. This makes models more performant, improves generalization, and can make learning more sample/data efficient.

Within genome assembly, we operate on input overlap graphs. By studying the symmetries of graphs by inspecting their invariances and equivariances, we are led to the @gnn machine learning architecture that is tailored to operate effectively on graph-structured data.
=== Permutation Invariance and Equivariance
Let $G = (V, E)$ be a graph such that $V$ is the set of nodes representing arbitrary entities. $E subset.eq V times V$ is the set of edges such that $(u, v) in E$ encodes relationships among these nodes/entities. The complete connectivity of $G$ has an algrebric representation $bold(A) in RR^(|V| times |V|)$, the adjacency matrix such that:
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

where $psi$ and $phi.alt$ are learnable message, and update functions, respectively, while $plus.circle$ is a permutation-invariant aggregation function (e.g., sum, mean, max). A permutation-equivariant GNN layer $bold(F)$ is the local message passing function applied over all neighborhoods of $G$:
$ bold(F)(features, edgefeatures, adj) = mat(dash.em f(bold(x)_1, features_(neighborhood_1), edgefeatures_(neighborhood_1)) dash.em;
                               dash.em f(bold(x)_2, features_(neighborhood_2), edgefeatures_(neighborhood_2)) dash.em;
                               dots.v;
                               dash.em f(bold(x)_n, features_(neighborhood_n), edgefeatures_(neighborhood_n)) dash.em;) $

A @gnn consists of sequentially applied message passing layers.

=== Expressivity of Graph Neural Networks
Graphs $G_1$ and $G_2$ are considered isomorphic if they encode the same adjacency structure under some permutation of their nodes. Although @gnn:pl are powerful graph processing tools, they are unable to solve all tasks on a graph accurately.

A @gnn is able to distinguish two non-isomorphic graphs $G_1$ and $G_2$, if it maps them to differing graph embeddings (in $RR^d$, for some arbitrary dimension $d in NN$) i.e. $bold(h)_(G_1) eq.not bold(h)_(G_2)$. The ability to distinguish non-isomorphic graphs is important as without this capability, solving a task requiring discriminating between them is unachievable. This graph isomorphism problem is also challenging, we no known polynomial-time algorithm known for it yet.

The expressive power of a @gnn is assessed by the set of graphs that they can distinguish (mapping them to different embeddings if, and only if, the graphs are non-isomorphic). Formally, assume that the set of all @gnn:pl is given by the set $PP$, and the set of all graphs is given by the set $GG$. Now, further assume that $P_1, P_2 in GG$ are arbitrary @gnn:pl, and that the set of graphs distinguishable by $P_1$ and $P_2$ are $DD_(P_1), DD_(P_2) subset.eq GG$). We then define the expressive power partial ordering over $PP$, $prec.eq$ , as:
$
  P_1 prec.eq P_2 <==> DD_(PP_1) subset.eq DD_(PP_2)
$
and consequently, we also have $P_1 prec P_2 <==> DD_(PP_1) subset DD_(PP_2)$.

It has been proven that the @gnn formulation laid out in @section:gnn is at most as powerful at distinguishing non-topologically identical graphs as the @wl test (note the similarity to @gnn message passing in @eq:gnn_message_passing):

#let wl_test = [#set text(size: 0.9em)
#algorithm({
  import algorithmic: *
  Function("Weisfeiler-Lehman-Test", args: ([$G_1 = (V_1, E_1)$],[$G_2 = (V_2, E_2)$],), {
    Cmt[Note that this is more specifically the 1-@wl test]
    State[]
    Cmt[Assign identical starting colors to each node in both graphs]
    Assign[$forall u in V_1, thin thin thin thin c_(u, G_1)$][$c_0$]
    Assign[$forall v in V_2, thin thin thin thin c_(v, G_2)$][$c_0$]
    State[]
    While(cond: [colors are not stable], {
      Cmt[Update each node's color]
      Cmt[Note that $"HASH"$ is some color hashing function]
      State[]
      Assign[$forall u in V_1, thin thin thin thin c_(u, G_1)^(t)$][$"HASH"(c_(u, G_1)^(t - 1), {{c_(w, G_1)^(t - 1)}}_(w in cal(N)_u))$]
      State[]
      Assign[$forall v in V_2, thin thin thin thin c_(v, G_2)^(t)$][$"HASH"(c_(v, G_2)^(t - 1), {{c_(w, G_2)^(t - 1)}}_(w in cal(N)_v))$]
    })
    State[]
    If(cond: [${{c_(u, G_1)^(t)}}_(u in V_1) eq.not {{c_(v, G_1)^(t)}}_(v in V_2)$], {
      [#v(0.5em) return _not_ isomorphic]
    })
    Else({
      [return _possibly_ isomorphic]
    })
  })
})]
#wl_test

@gnn expressivity is an important topic for solving problems on graphs that require identifying and differentiating graph structure. Since the layout problem in genome assembly is fundamentally about graph structure, this is a critical area of interest.

== Mamba Selective State Space Model
Mamba is derived from the class of @s4 models @mamba, combining aspects of recurrent, convolutional, and classical state space models. While @s4 models have a recurrent formulation, a parallelizable convolutional operation applied to a sequence yields the identical result, making them much faster than previous @rnn architectures, such as @lstm networks.

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
    figure(image("graphics/s4_continuous.svg", height: 4cm), caption: subcaptionlong[S4 Cont.]), <fig:s4_continuous>,
    figure(image("graphics/s4_discrete_recurrent.svg", height: 4cm), caption: subcaption[S4 Discrete (Recurrent)]), <fig:s4_discrete_recurrent>,
    figure(image("graphics/s4_discrete_convolutional.svg", height: 4cm), caption: subcaption[S4 Discrete (Convolutional)]), <fig:s4_discrete_convolutional>,
    caption: [Illustration of the continuous-time @s4 model in (a). (b) and (c) show how the discrete @s4 model can be represented equivalently by recurrence and convolution.]
  )]

Moreover, these models have principled mechanisms for long-range dependency modelling @ssm-long-range, and perform well in benchmarks such as Long Range Arena @long-range-arena. Their speed and ability to capture long-range dependencies make them compelling for sequence modelling tasks.

Formally, S4 models, defined with continuous-time parameters $(Delta, bold(A), bold(B), bold(C)))$ can be formulated as follows:
$ h'(t) = bold(A) h(t) + bold(B) x(t) $
$ y(t) = bold(C) h(t) $

These equations refer to a continuous-time system, mapping a _continuous _ sequence $x(t) in bb(R) arrow.r y(t) in bb(R)$, through an implicit hidden latent space $h(t) in bb(R)^N$ (illustrated in @fig:s4_continuous). For discrete data however, like a sequence of bases in a read, these equations need to be discretized. Before detailing the discretization procedure, we note that having an underlying continuous-time system is beneficial as we inherit beneficial properties of continuous-time dynamics---key is smoother encoding of long-range dependencies and memory. Moreover, there are well-established connections between discretization of continuous time systems and @rnn gating mechanisms.


Discretization is performed using the step size parameter $Delta$, transforming the continuous-time parameters $(Delta, bold(A), bold(B))$ into discrete-time parameters $(bold(dash(A)), bold(dash(B)))$ through a discretization rule. $Delta$ can be viewed as a more generalized version of the gating mechanism found in @rnn:pl. Mamba Selective State Space model uses zero-order hold as its discretization rule, where $dash(A) = exp(Delta A)$ and $dash(B) = (Delta A)^(-1)(exp(Delta A) - I) dot Delta B$). This yields a new set of discrete equations:
$ h_t = dash(bold(A))h_(t - 1) + dash(bold(B)) x_t $
$ y_t = bold(C)h_t $

Through repeated application of the recurrence relation, and simplification via the @lit property (which states that $(Delta, bold(A), bold(B))$ and consequently $(bold(dash(A)), bold(dash(B)))$ remain constant for all time-steps), the system can be equivalently expressed as a 1-dimensional convolution (see @fig:s4_discrete_recurrent and @fig:s4_discrete_convolutional for illustration) over the sequence $x$ with kernel $bold(dash(K))$ ($star$ denotes the covolution operation):
$ bold(dash(K)) = (C dash(B), C dash(A) dash(B), ..., C dash(A)^k dash(B), ...) $
$ y = x star bold(dash(K)) $

Since @s4 models have fixed parameters with respect to the inputs, they cannot perform content-based reasoning, essential for tasks such as language, or genome modelling. To address this, Mamba extends the S4 formulation by incorporating _selectivity_---the ability to select data in an input-dependent manner, helping filter out irrelevant data, and keep relevant information indefinitely, by making the parameters functions of the input. However, this breaks the time- and input-invariance (@lit) that allows fast convolution-based calculation. This is compensated for by replacing the convolution with a scan/prefix sum operation (@fig:recurrent and @fig:parallel_scan show how the scan/prefix sum algorithm produces the same result as recurrence). 

The ability to select data in an input-dependent manner, along with the scan/prefix sum in-place of convolution, together results in the Mamba _Selective_ State Space Model (henceforth referred to simply as Mamba) (@fig:mamba_official shows the Mamba diagram from the original paper).

#place(top + center)[#subpar.grid(
  columns: 2,
  gutter: 3em,
  figure(image("graphics/recurrent.svg", height: 8cm, fit: "contain"), caption: subcaptionlong[Recurrent formulation for generating @s4 hidden states]), <fig:recurrent>,
  figure(image("graphics/parallel_scan.svg", height: 8cm, fit: "contain"), caption: subcaptionlong[Scan/Prefix sum formulation]), <fig:parallel_scan>,
  caption: [Illustration of how the scan/prefix sum algorithm produces the same result as the recurrent (sequential) formulation in generating the @s4 hidden states in parallel. Note that, for example, $x_3$'s calculation begins before $x_2$ has been fully calculated.]
)]

#figure(image("graphics/mamba_official.png"), caption: [Mamba's selection mechanism in #text(fill:blue.darken(50%))[blue] alters parameters in an input-dependent manner.]) <fig:mamba_official>


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
Ultra-long reads have have demonstrated significant advantages in resolving complex artifacts in overlap graphs and repeating regions in genomes, and prior work has presented @gnn:pl as a viable method for improving the layout phase in the @olc algorithm. We are interested in investigating the utility of @gnn:pl in leveraging ultra-long read data to advance the capabilities of neural genome assembly methods.

The integration of ultra-long reads with conventional long-read data alters the structural properties of the resulting overlap graphs. This motivates exploring alternative @gnn architectures that may better exploit the additional information available. In this chapter, we detail our training and inference setup, discuss the various @gnn architectures tested, and explain our integration method of raw read data into the model.
#place(top + center)[#figure(
  image("graphics/overview.svg"),
  caption: [(A) Simulated @pacbio @hifi reads are generated from a reference genome via PBSIM. Additional @ont @ul reads may also be generated. Alternatively, real read data can also be provided. The reads are then passed to Hifiasm, which constructs the corresponding overlap graph. (B) Ground-truth edge labels are computed corresponding to the optimal assembly. During training only, the overlap graph is masked and partitioned. Masking allows for data augmentation by simulating varying read coverage from $times 30$ to $times 60$. Partitioning is required to fit onto @gpu memory. (C) Features are extracted from the overlap graph according to the model used, and edge probability predictions are made by the model. Note the reversed compliment is only used by some of the @gnn models. The loss in computed relative to the ground-truth labels. (D) The genome is reconstructed via greedy decoding.]
) <fig:overview>]

== Training and inference setup
We detail the training and inference pipeline next. A detailed illustration can be found in @fig:overview.

=== Generating the overlap graph
The first step of generating an overlap graph is gathering the raw read data. Since we are unable to produce our own sequencing data, reads from the CHM13v2 @t2t human genome assembly are instead simulated. This simulation is performed using a utility called PBSIM3 that emulates the read profile of @pacbio @hifi long-reads according to fastq data (fastq is a format for storing the sequencing data, in addition to per-base quality scores that are crucial for our simulation) from the sequencing of the HG002 draft human reference. When simulating reads, a $times #h(0em) 60$ coverage factor is used (enough reads to cover the genome $60$ times over).

For training, we choose chromosomes 19 and 15, representing both non-acrocentric, and acrocentric chromosomes (an acrocentric chromosome is one where the centromere, the region of a chromosome that holds sister chromatids together, is not located centrally in the chromsome, but towards one end). For validation and test, we likewise choose chromsomes 11 and 22, and chromosomes 9 and 21, respectively. Note that the chromosomes chosen represent the most difficult ones during assembly due to the tangles often present in their real-life overlap graphs. Additionally, the centromeric region of each of these chromosomes is extracted for generating reads, where most complexity arises. By training on only a small portion of the chromosomes present in the genome, we demonstrate the positive generalization capabilities of our neural method.

Once the reads are generated, Hifiasm, a de novo assembler specifically designed for @pacbio @hifi read data, is used to generate the overlap graph. Note that no traditional graph simplification algorithms like transitive edge removal, dead-end trimming, or bubble removal, are applied. Also, it is important to note that the overlap graph produced is a symmetric overlap graph as the reads can belong to either strand of the @dna. The symmetric overlap graph consists of one graph, and its dual that contains a duplicate set of nodes representing the same reads, but with the edges reversed. This is due to an interesting property during sequencing where reads from the dual @dna strand are sequenced in reverse order along the length of the @dna.

=== Overlap graph ground-truth label generation
We refer to whether an edge belongs to the finally assembly as a boolean _label_, which is the target the @gnn aims to predict. There are two conditions for an edge to be valid (i.e. labeled true): (1) The reads the edge states overlap must be sampled from the same strand of @dna (@eq:same_strand) and have a valid overlap (@eq:valid_overlap), and (2) the edge must not lead to a read that is a dead-end. Formally, the first condition states that for reads $A$ and $B$, with edge $A -> B$:
$
A_"strand" = B_"strand" "(same strand)"
$ <eq:same_strand>
#set math.cases(reverse: true)
$
cases(A_"start" &< B_"start",
A_"end" &> B_"start",
A_"end" &< B_"end") "(valid overlap)"
$ <eq:valid_overlap>

where $X_"strand"$, $X_"start"$, $X_"end"$ refer to the strand, starting, and ending positions in the actual genome for some read $X$. Edges not satisfying this first condition are marked with the label false. Note that since the reads are simulated, we know the true strand and positions along the genome they are sampled from. To find the edges also satisfying the second property, we follow the algorithm laid out below:
#let algorithm_1 = [#set text(size: 0.9em)
#algorithm({
  import algorithmic: *
  Function("Find-Optimal-Assembly", args: ([_overlap-graph_],), {
    Cmt[Initialize the set of edges belonging to the optimal assembly]
    Assign[_optimal-edges_][${}$]
    State[]
    For(cond: [_connected-component_ *in* _overlap-graph_], {
      Cmt[Decompose the connected-component into nodes and edges]
      Assign[$V$, $E$][_connected_component_]
      State[]
      Cmt[Start search from the read at the lowest position along the genome sequence]
      Assign[_lowest-read-node_][$"argmin"_(v thin in thin V)$ #FnI[get-read-start-loc-for-node][$v$]]
      State[]
      Cmt[Perform the forward @bfs]
      Assign[_visited-nodes-forward_][${}$]
      Assign[_visited-edges-forward_][${}$]
      Assign[_visited-nodes-forward_, _visited-edges-forward_][#linebreak() #h(2em) #FnI[@bfs][_connected-component_, start=_lowest-read-node_]]
      State[]
      Cmt[Start the reverse search from the _visited_ read at the highest position]
      Assign[_highest-read-node_][$"argmax"_(v thin in italic("visited-nodes-forward"))$ #FnI[get-read-start-loc-for-node][$v$]]
      State[]
      Cmt[Perform the reverse @bfs]
      Assign[_visited-nodes-backward_][${}$]
      Assign[_visited-edges-backward_][${}$]
      Assign[_visited-nodes-backward_, _visited-edges-backward_][#linebreak() #h(2em) #FnI[@bfs][_connected-component_, start=_highest-read-node_]]
      State[]
      Cmt[The edges belonging to the final assembly are traversed by both @bfs:pl]
      Assign[_optimal-edges_][_optimal-edges_ $union$ (_visited-edges-forward_ $inter$ _visited-edges-backward_)]
    })
    State[]
    Return[_optimal-edges_]
  })
})]
#algorithm_1

We start from the edge whose starting read is at the lowest position along the genome and perform a @bfs from it, storing the visited nodes. From this set of visited nodes, another @bfs is performed starting from the node representing the read at the highest genomic position. Edges traversed by both of the @bfs:pl belong to the optimal assembly (called _optimal-edges_). If there are multiple connected components in the overlap graph, the process is repeated. The _optimal-edges_ are labeled as belonging to the final assembly (true)---all other edges are labeled false.

=== Overlap graph masking and partitioning
Masking and partitioning is performed during training only, with the entire graph used for performing inference. Masking is performed as a form a data augmentation to cheaply produce different sets of reads and the corresponding overlap graph. For every training step, $0$--$20%$ of the overlap graph's nodes, and corresponding edges, are removed. This simulates varying levels of read coverage up to the original $times #h(0em) 60$.

Additionally, since the entire overlap graph contains $>100,000$ nodes, and cannot fit onto @gpu memory, METIS partitioning is used to divide the overlap graph. Note that inference is performed on the @cpu, which is able to access the system's main memory, and so graph partitioning is not required.

=== Feature extraction and running the models
We leave discussion of the node and edge features extracted from the overlap graph and raw read data to later sections, since they depend on the model architecture used. The models then take the overlap graph, and these node and edge features as input, producing for each edge, a probability of that edge belonging to the final assembly.

=== Reconstructing the genome via greedy decoding
Once a probability has been assigned to each edge representing its likelihood of belonging to the final assembly, we apply a greedy decoding algorithm (detailed below) to extract contigs---sets of overlapping @dna fragments that together reconstruct a contiguous portion of the genome:

#let algorithm_2 = [#set text(size: 0.9em)
#algorithm({
  import algorithmic: *
  Function("Greedy-Decode-Contigs", args: ([_overlap-graph_],[_edge-probabilities_]), {
    Assign[_final-assembly_][${}$]
    State[]
    While(cond: [_overlap-graph_ contains unvisited nodes], {
      Cmt[Sample $B$ starting edges using an empirical distribution given by _edge-probabilities_]
      Assign[$E$][${e_1, ..., e_B}$, where $e_i$ is an edge with probability $bb(P)(e_i) = italic("edge-probabilities")[$e_i$]$]
      State[]
      For(cond: [$e_i in E$], {
        Cmt[Initialize path greedily decoded from this edge]
        Cmt[Note that in this pseudocode although the path $p_i$ is a list of edges, we also allow for checking if a node is in the path for ease of notation]
        Assign[$p_i$][[$e_i$]]
        State[]
        Cmt[Greedy forward search from $v_i$ (target node of $e_i: u_i -> v_i$)]
        Cmt[During greedy forward search, the new edge to be traversed must be unvisited, and lead to an unvisited node]
        While(cond: [unvisited outgoing edge from last node in $p_i$, $v_k$, exists], {
          Cmt[Choose outgoing edge from $v_k$ with highest probability]
          Assign[$e_k$][$"argmax"_("outgoing edge" e_k "from" v_k) bb(P)(e_k) $]
          Cmt[Append this edge to extend the path $p_i$]
          Assign[$p_i$][$p_i$ + [$e_k$]]
        })
        State[]
        Cmt[Greedy backward search from $u_i '$ (virtual pair of source node $u_i$ of $e_i: u_i -> v_i$)]
        Cmt[During greedy backward search, the new edge to be traversed must be unvisited, and its source must be an unvisited node]
        While(cond: [unvisited incoming edge to first node in $p_i$, $v_j$, exists], {
          Cmt[Choose incoming edge from $v_j$ with highest probability]
          Assign[$e_j$][$"argmax"_("incoming edge" e_j "from" v_j) bb(P)(e_j) $]
          Cmt[Prepend this edge to extend the path $p_i$]
          Assign[$p_i$][[$e_j$] + $p_i$]
        })
        State[]
        Cmt[Mark transitive nodes as visited]
        For(cond: [node $v in.not p_i$], {
          If(cond: [
            #FnI[predecessor][$v$] $in p_i and$ #FnI[successor][$v$] $in p_i $ #linebreak() $and e:$ #FnI[predecessor][$v$] $->$ #FnI[successor][$v$] $in p_i$
          ], {
            FnI[mark-node-visited][$v$]
          })
        })
      })
      State[]
      Cmt[Keep the longest path]
      Assign[_longest-path_][$"argmax"_(p_i)$ #FnI[length][$p_i$]]
      State[]
      Cmt[Convert the set of reads in the _longest-path_ into a contig]
      Assign[_contig_][#FnI[to-contig][_longest-path_]]
      State[]
      Cmt[Add the _contig_ to the _final-assembly_]
      Assign[_final-assembly_][_final-assembly_ $union$ _contig_]
      State[]
      Cmt[The nodes (and edges) used to form the contig cannot be reused to avoid duplicating regions]
      State[#FnI[mark-nodes-visited][_longest-path_]]
      State[]
      Cmt[Stop when the length of the longest contig found falls below a fixed threshold]
      If(cond: [#FnI[length][_longest_path_] $<$ _min-contig-length_], {
        State[break]
      })
    })
    State[]
    Return[_final-assembly_]
  })
})]
#algorithm_2

Recall that we are interested in finding a Hamiltonian path through the overlap graph to recover the genome. In an ideal scenario, where all neural network edge predictions are accurate and the graph contains no artifacts, a simple greedy traversal (forwards and backwards) starting from any positively predicted edge would suffice to reconstruct the genome. However, due to prediction errors and noise in the graph, neither of these conditions are met in practice, and so we use a greedy decoding algorithm.

This algorithm first samples multiple high-probability seed edges and then greedily chooses a sequence of edges both forwards and backwards from each seed edge, forming a path through the assembly graph. The longest resulting path is selected and overlapping reads along that path merged into a contig. Nodes along the selected path are marked as visited to prevent their reuse in subsequent searches, and the process repeats until no path above a fixed length threshold can be found.

== Model architectures
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
The embedding of the standard input features into the initial hidden representations $h_i^0 in bb(R)^d$ for node $i$ at layer $0$, and $e_(s t)^0 in bb(R)^d$ for the edge $s -> t$ (where $s$ and $t$ are nodes) at layer 0 are computed as:
$
  h_i^0 &= W_2^"n" (W_1^"n" x_i + b_1^"n") + b_2^"n" \
  e_(s t)^0 &= W_2^"e" (W_1^"e" z_(i j) + b_1^"e") + b_2^"e"
$
where all $W^"n"$ and $b^"n"$, and $W^"e"$ and $b^"e"$ represent learnable parameters for transforming the node and edge features respectively ($W_1^"n", W_1^"e" in bb(R)^(d times 2)$, $W_2^"n", W_2^"e" in bb(R)^(d times d)$, and $b_1^"n", b_1^"e", b_2^"n", b_2^"e" in bb(R)^d$), and $d$ is the hidden dimension.

We refer to this formulation for $h_i^0$ and $e_(s t)^0$ as the standard input embedding.

=== SymGatedGCN
Let the hidden representations of node $i$ and edge $e_(s t): s -> t$ at layer $l$ be $h_i^l$ and $e_(s t)^l$ respectively. Additionally, let $j$ denote node $i$'s predecessors and $k$ denote its successors. Each @symgatedgcn layer then transforms the hidden node and edge embeddings as follows:
#let relu = [$"ReLU"$]
#let norm = [$"Norm"$]
$
  h_i^(l + 1) = h_i^l + #relu (#norm (A_1^l h_i^l + sum_(j -> i) eta_(j i)^("f", l + 1) dot.circle A_2^l h_j^l + sum_(i -> k) eta_(i k)^("b", l + 1) dot.circle A_2^l h_j^l))
$
$
  e_(s t)^(l + 1) = e_(s t)^l + #relu (#norm (B_1^l e_(s t)^l + B_2^l h_s^l + B_3^l h_t^l))
$ <eq:edge_features>
where all $A, B in RR^(d times d)$ are learnable parameters with hidden dimension $d$, #relu stands for Rectified Linear Unit, and #norm refers to the normalization layer used---this is discussed in more detail in @sec:granola. Note that the standard input embeddings (@sec:standard_input_embedding) are used for $h_i^0$ and $e_(s t)^0$. $eta_(j i)^("f", l)$ and $eta_(i k)^("b", l)$ refer to the forward, and backward gating functions respectively. The edge gates are defined according to the GatedGCN:
$
  eta_(j i)^("f", l) = sigma (e_(j i)^l) / (sum_(j' -> i) sigma (e_(j' i)^l) + epsilon.alt) in [0, 1]^d, #h(2.5em) eta_(i k)^("b", l) = sigma (e_(i k)^l) / (sum_(i -> k') sigma (e_(i k')^l) + epsilon.alt) in [0, 1]^d
$
where $sigma$ represents the sigmoid function, $epsilon.alt$ is a small value added to prevent division by 0, and $j' -> i$ represents all edges where the destination node is $i$. Likewise, $i -> k'$ represents all edges where the source node is $i$.

#let modelexplanation = it => [
  #box(fill: blue.lighten(90%), inset: 1em, stroke: blue, radius: 1em, width: 100%)[#it]
]

#modelexplanation[
  Most conventional @gnn layers are designed to operate on undirected graphs, and therefore do not account for directional information intrinsic to overlap graphs. This limitation is problematic, since the overlap graph encodes the directional path reflecting the linear structure of the genome from start to end. @symgatedgcn aims to address this lack of expressivity by distinguishing the messages passed along the edges $(sum_(j -> i) eta_(j i)^("f", l + 1) dot.circle A_2^l h_j^l)$, to those passed along the reversed direction of the edges $(sum_(i -> k) eta_(i k)^("b", l + 1) dot.circle A_2^l h_j^l)$.
]

=== GAT+Edge
The standard @gat architecture only focusses on node features, and so we extend this architecture to update edge features, include them in the attention calculation, and use them to also update the node features.

First, updated edge features are calculated identically to @symgatedgcn (@eq:edge_features).

In contrast to the @gat architecture with a single shared attention mechanism, there are now two mechanisms, $a^"n"$ and $a^"e"$, which compute the attention coefficients for nodes and edges respectively ($a^"n", a^"e": RR^d times RR^d times RR^d -> RR$). Each mechanism is implemented via separate, single-layer feed-forward neural networks. The attention coefficients are given as follows:
$
  c_(i j)^"n" &= a^"n" (h_j^l || e_(j i)^l || h_i^l) \
  c_(i j)^"e" &= a^"e" (h_j^l || e_(j i)^l || h_i^l) \
$
where $c_(i j)^"n"$ indicates the importance of node $j$'s features to node $i$, and $c_(i j)^"e"$ indicates the importance of the edge $e_(j i): j -> i$ to node $i$. $||$ denotes the concatenation operator along the hidden dimension. These coefficients are then normalized over all $j$ to make them comparable across nodes, via softmax:
$
  alpha_(i j)^"n" &= "softmax"_j (c_(i j)^"n") = (exp (c_(i j)^"n")) / (sum_(k in neighborhood_i) exp (c_(i k)^"n")) \
  alpha_(i j)^"e" &= "softmax"_j (c_(i j)^"e") = (exp (c_(i j)^"e")) / (sum_(k in neighborhood_i) exp (c_(i k)^"e")) \
$
The updated node features are then calculated by first weighing the node and edge features by their corresponding normalized attention coefficients. Next, these node and edge features are concatenated, and passed through another single-layer feed-forward neural network, $"mix-node-edge-information"$:
$
  
  h_i^(l + 1) = "mix-node-edge-information"(lr(sigma (sum_(j in neighborhood_i) alpha_(i j)^"n" bold(W)^"n" h_j^l ) ||) thin sigma (sum_(j in neighborhood_i) alpha_(i j)^"e" bold(W)^"e" e_(j i)^l ))
$
where $bold(W)^"n", bold(W)^"e" in RR^(d times d)$ are parameterized weight matrices.

#modelexplanation[
  We refer to our custom attention-based formulation, which incorporates edge features, as GAT+Edge. This architecture extends the original @gat by not only implicitly enabling assignment of different importances to nodes of the same neighborhood, but also across edges. Importantly, this addresses a key limitation of the standard @gcn architecture, but note that this is mitigated with the gating mechanism introduced with GatedGCN. Additionally, GAT+Edge remains a computationally efficient architecture.
]

=== SymGAT+Edge
With the design of this architecture, we aim to combine the symmetry feature from @symgatedgcn with the GAT+Edge architecture mentioned previously. This is done by first calculating the updated edge features $e_(s t)^(l + 1)$ identically to @symgatedgcn (@eq:edge_features).

Next, a copy of the input graph $G = (V, E)$ is made, $G_"rev" = (V, E_"rev")$, such that:
$ forall i, j in V. thick i -> j in E <==> j -> i in E_"rev" $
$G_"rev"$ is equivalent to the original graph $G$, with the direction of all edges reversed. GAT+Edge then individually takes $G$ and $G_"rev"$ as input, producing a pair of new node features $h_i^("f", l + 1)$ and $h_i^("b", l + 1)$ respectively. These are then combined to produced to new hidden node state as follows:
$
  h_i^(l + 1) = h_i^l + #relu (#norm h_i^("f", l + 1) + h_i^("b", l + 1) )
$

#modelexplanation[
  Integrating the symmetry feature from @symgatedgcn into GAT+Edge, to form SymGAT+Edge, helps to increase expressivity as messages passed along edges cannot be distinguished from messages passed along the reversed direction by the attention mechanism either. 
]

=== SymGatedGCN+Mamba
The standard input features (@sec:standard_input_features) used in prior work on neural genome assembly extract normalized overlap length and similarity from pairs of overlapping reads. However, the models have access to only these summary statistics, not the raw nucleotide read data, which could enable the model to extract more complex features, for example by capturing some notion of what is biologically plausible.

While encoder-only Transformers are the contemporary choice for sequence-to-embedding tasks like this, a fundamental drawback makes them unsuitable---their quadratic complexity with respect to the sequence length. Each read is upto $10s$ of @kb long for @pacbio @hifi reads, and there are $1000$s of reads even in the partitioned overlap graph used during training (note that we cannot partition the graph to an arbitrarily small number of nodes without sustaining major losses in performance as context around the graph artifact is lost).

On the other hand, @rnn architectures such as @lstm have linear complexity, but have traditionally struggled with modelling such long sequences. The key to the efficacy of Transformers, is the self-attention mechanism's ability to effectively route information from across the sequence, regardless of the distance.

As a result, we turn to the Mamba architecture, which with its selectivity mechanism and parallel scan implementation, is able to model complex, long sequences, without the computational cost of Transformers.

Additionally, another issue mitigated by the use of Mamba is that there is no canonical tokenization for a sequence of nucleotides. Operating directly on the nucleotide sequence is important for de novo sequencing, where we have no knowledge of the underlying genome, due to the absence of a reference. The Mamba model has been previously shown to operate well directly on nucleotide sequences on tasks involving @dna modelling.

The SymGatedGCN+Mamba model uses the standard input features (from @sec:standard_input_features) in addition to the Mamba encoding of the reads as additional node features. Assume we are given an overlap graph $G = (V, E)$. For read $r_i$, represented by node $v_i$, the Mamba read encoding node feature $m_i in bb(R)^d$ is generated as follows ($d$ is size of the hidden dimension).

First, read $r_i in {"A, T, C, G"}^n$, which is a string of nucleotides of length $n$, is one-hot encoded to produce $r_i^"one-hot" in {0, 1}^(n times 4)$:
$
  r_(i, j)^"one-hot" = cases(
    (0, 0, 0, 1) "if " r_(i j) = "A",
    (0, 0, 1, 0) "if " r_(i j) = "T",
    (0, 1, 0, 0) "if " r_(i j) = "C",
    (1, 0, 0, 0) "if " r_(i j) = "G",
  )
$
where $j$ refers to the $j$th nucleotide in $r_i$. Next, the one-hot encoded representation is expanded to the hidden dimension $d$ via a learned parameter matrix $bold(W)^"expand" in 4 times d$, and then the read is encoded into $r_i^"encoded" in n times d$ by Mamba:
$
  r_i^"encoded" = "Mamba"(r_i bold(W)^"expand")
$
Note that $r_i^"encoded"$ is a matrix that varies in size with the length of the read. In order to obtain a fixed length hidden encoding of the read, we simply take the last row of this matrix (indexing from 1):
$
  m_i = r_i^"encoded" [n]
$

The initial node and edge hidden embeddings are then given by:
$
  h_i^0 &= W_2^"n" (W_1^"n" (x_i || m_i) + b_1^"n") + b_2^"n" \
  e_(s t)^0 &= W_2^"e" (W_1^"e" z_(i j) + b_1^"e") + b_2^"e"
$
where all $W^"n"$ and $b^"n"$, and $W^"e"$ and $b^"e"$ represent learnable parameters for transforming the node and edge features respectively ($W_1^"n" in bb(R)^(2 + d times d), W_1^"e" in bb(R)^(d times 2)$, $W_2^"n", W_2^"e" in bb(R)^(d times d)$, and $b_1^"n", b_1^"e", b_2^"n", b_2^"e" in bb(R)^d$), and $d$ is the hidden dimension. $||$ denotes the concatenation operator.

#modelexplanation[
  The primary goal of SymGatedGCN+Mamba is to explore whether the model can exploit the raw read data to generate new (node) features that are useful in resolving overlap graph artifacts. Mamba was chosen as the read encoding model of choice due to its near-linear time complexity, long-range dependency modelling capabilities, and promising results on adjacent @dna modelling tasks.
]

=== SymGatedGCN+MambaOnly
We use the same Mamba read encoding node feature $m_i in bb(R)^d$ as in SymGatedGCN+Mamba, but remove the dependency on standard edge features (@sec:standard_input_features). The initial node and edge embeddings are now given as:
$
  h_i^0 &= W_2^"n" (W_1^"n" (x_i) + b_1^"n") + b_2^"n" \
  e_(s t)^0 &= W_2^"e" (W_1^"e" (m_i || m_j) + b_1^"e") + b_2^"e"
$
where all $W^"n"$ and $b^"n"$, and $W^"e"$ and $b^"e"$ represent learnable parameters for transforming the node and edge features respectively ($W_1^"n" in bb(R)^(d times 2), W_1^"e" in bb(R)^(d times 2d)$, $W_2^"n", W_2^"e" in bb(R)^(d times d)$, and $b_1^"n", b_1^"e", b_2^"n", b_2^"e" in bb(R)^d$), and $d$ is the hidden dimension. $||$ denotes the concatenation operator.

#modelexplanation[
  SymGatedGCN+MambaOnly tests whether the model can recover the overlap length and similarity metrics used earlier, from raw read data (or alternatively generate even richer embeddings).
]

=== Graph Adaptive Layer Normalization <sec:granola>
Normalization has been shown to be critical for enhancing the training stability, convergence behavior, and overall performance of neural networks. Conventional normalization techniques such as BatchNorm, InstanceNorm, and LayerNorm, have been widely adopted, but are not specifically designed to support graph-structured data. In fact, direct application of these standard normalization techniques can impair the expressive power of @gnn:pl, degrading performance significantly.

Increasing the depth of a @gnn by stacking additional layers theoretically expands the class of functions it can represent, but repeated message passing operations can lead to node embeddings becoming indistinguishable---an effect known as over-smoothing. This observation is also theoretically motivated---graph convolution can be viewed as a type of Laplacian smoothing, and so its repeated applications in @gnn layers leads to embeddings having similar values. Mitigating over-smoothing is the primary motivator for many graph-specific normalization layers (e.g. PairNorm and DiffGroupNorm).

Unfortunately, despite numerous efforts to develop graph-based normalization schemes, no method consistently outperforms the alternatives in all tasks and benchmarks. Furthermore, normalization schemes extended from traditional schemes such as BatchNorm and InstanceNorm, often reduce the expressive power of the @gnn. The @granola authors postulate that there are two main reasons why these alternative schemes do not provide an unambiguous performance improvement across domains. Firstly, many methods use shared affine normalization parameters across all graphs, failing to adapt to input graph-specific characteristics. Secondly, special regard should be given to the expressive power of the normalization layer itself.

Since commonly used @gnn architectures are at most as powerful as the @wl graph isomorphism heuristic, any normalization layer designed using them will be unable to distinguish all input graphs, and therefore will fail to adapt the normalization parameters correctly to suit the input. More expressive architectures such as $k$-GNNs, whose design is motivated by the generalization of 1-@wl to $k$−tuples of nodes ($k$-WL), are accompanied by unacceptable computation and memory costs (e.g. $cal(O)(|V|^k)$ memory for higher-order MPNNs, where $V$ is the number of nodes in the graph).

@rnf is an easy to compute (and memory efficient), yet theoretically grounded alternative technique involving concatenating a different randomly generated vector to each node feature. This simple addition not only allows distinguishing between 1-@wl indistinguishable graph pairs based on fixed local substructures, but @gnn:pl augmented with @rnf are provably universal (with high probability), and thus can approximate any function defined on graphs of fixed order. 

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

Under each model architecture, have a box for why there is that change

At the beginning of this entire section, story of we're trying to integrate extra ultra-long data -> might require better gnns
  - But, in evaluation, since these new gnns are at the same expressivity level in the wl heirarchy, and this is a very structural problem, there is no benefit



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