# Reference authenticity and use audit

Audit date: 2026-09-05

## Outcome

All 35 references in `paper/references.tex` resolve to matching records in
Crossref or arXiv. No fabricated title, author list, venue, or DOI was found.
The machine-readable record is `reference_registry_audit.json`; it contains the
queried metadata, source title, title-similarity result, authors, venue, year,
volume, issue, pages, DOI/URL, and lookup status for every bibliography item.

Final automated result: **35 verified, 0 mismatches, 0 lookup failures**.

## Corrections made

1. `ref1` (Stern et al.) had an end page of 159. The official SoCS issue and
   DOI record give pages 151--158. The page range was corrected and DOI
   `10.1609/socs.v10i1.18510` was added. The bibliography retains 2019 because
   this is the date of the original Vol. 10 No. 1 SoCS 2019 issue; the migrated
   OJS/Crossref record displays a 2021 platform publication date.
2. `ref32` (Sturtevant) was bibliographically correct but omitted its DOI.
   DOI `10.1109/TCIAIG.2012.2197681` was added.

## Registry coverage

- Crossref DOI lookup: refs. 1--8, 11, 13--35 except the three arXiv-only
  records below.
- arXiv API: refs. 9 (`2101.03805`), 10 (`2108.00745`), and 12
  (`2211.12885`). Their titles, authors, identifiers, and initial years match.
- Official-page spot checks were also performed for the SoCS MAPF definition
  paper, probabilistic robust MAPF, fault-tolerant offline MAPF, the 2025 Xu et
  al. shortest-path paper, the 2025 all-optimal-MAPF paper, and the IEEE Access
  NSGA-III/application papers.

## Citation-use corrections

The audit also checked whether nearby prose matched what each cited work
actually studies. The manuscript was revised as follows:

- multi-robot task planning/allocation papers are described as related
  applications, not MAPF solver variants;
- the contingent-planning citation is described as execution uncertainty, not
  specifically as an edge-closure method;
- the Nemhauser et al. paper is cited at the submodular greedy guarantee rather
  than grouped with archive-performance studies;
- the signed-distance-field exploration paper is described as separate safe
  path-planning work, not as an NSGA-III application;
- the Xu adaptations remain explicitly mechanism-level controls rather than a
  claim of reproducing the full published evolutionary algorithm.

## Scope of assurance

This audit establishes bibliographic authenticity and checks the manuscript's
immediate use of each citation. It does not claim to reproduce every cited
paper's experiments. The local JSON is a dated audit artifact; DOI publisher
pages remain the canonical records.
