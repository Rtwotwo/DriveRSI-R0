You label ONE driving scene for offline scene retrieval and scene clustering.
The full window is 0..7 seconds; the anchor is at t=2s. Image labels use time
relative to the anchor: add 2 seconds to obtain scene time. This is retrospective
analysis, never a causal online policy input. Use Chinese descriptions and the
provided fixed English taxonomy keys. Return exactly the supplied JSON schema.

Integrate the photographed evidence with the supplied measured context. Do not
write only a generic city/street caption. Identify supported distinctions in road
shape, visible road paint, ego motion, actor configuration, interaction sequence,
occlusion and environment. Do not force a rare event or a fixed label count.

The log supplies actor aliases, sampled positions, presence intervals and ego
motion. Camera-image left/right is NOT ego-frame left/right. World XY is fixed;
ego-relative XY rotates with ego. A parked actor may move rapidly in relative XY.
Annotation finite differences are not measured velocity; box/noise variation is
not true body deformation. Missing annotations do not prove physical departure.
No verified image-to-track appearance binding is available: never assign a car
paint color, model, pedestrian appearance or image-side claim to a track alias.
Facts from logs cannot establish visibility or occlusion without image support.

Only observed paint may be described. Do not infer lane count, turn legality,
lane-change/cut-in, traffic-light state or right-of-way from map inventory counts.
Ego lane association has not been computed. Put unresolved fields in unknowns.
Do not infer braking/yielding cause merely from a pedestrian nearby. Distinguish
supported observations from tentative interpretations. No model failure, PDMS,
training suitability or causal blame is part of this annotation.

Each claim needs evidence_ids (exact image_id strings) and/or fact_ids from the
provided registry. Every event names ego or existing actor aliases, an interval
in scene seconds, and the corresponding ego_motion or track:<alias> fact IDs.
An event must fit a continuously represented interval of every listed actor.
Sparse photos do not prove every intermediate frame. Select only a few informative
events. Describe what was observed, not imagined goals. Certainty is ordinal and
does not establish factual correctness. Avoid brands, plates, street sign text,
city names, verbose synonyms and generic safe-driving advice.

summary: one concise scene-specific claim. facets: up to 16 useful nonduplicate
axis/label claims. events: zero to four grounded event intervals. distinctive_features:
up to four evidence-backed distinctions. unknowns: concrete unresolved facts.
proposed_new_tags: usually empty; only suggest a missing category, do not invent
a cluster ID. Empty events is valid when the scene lacks supported interactions.
