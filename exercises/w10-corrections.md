# W10 Written Drill - Review and Corrections

Reviewed on: 2026-11-03.
Original: [Handwritten SVM drill](w10-first-attempt.pdf), two pages, archived in commit `358a662`.
Reference: HCMUT SVM slides, 11 August 2026, slides 5-13 and 34-35. The original scan is unchanged.

## Q1. Maximum margin and support vectors

The hard-margin objective and constraints are correct:

$$\min_{w,b}\frac12\lVert w\rVert^2,
\qquad t_i(w^Tx_i+b)\geq1.$$

**Notation clarification:** the answer calls 2/||w|| the geometric margin. This is the full width between the two supporting planes. The lecture defines the margin as the shortest distance from the decision boundary to a training point, which is 1/||w|| in canonical scaling. Both widths are maximized by minimizing the squared norm, so the optimization reasoning is correct.

The description of support vectors on the two supporting planes is appropriate for this hard-margin question. For soft margin, support vectors can also lie inside the margin or be misclassified. The broader statement is that they have nonzero dual multipliers.

## Q2. Geometric example

All calculations are correct. The boundary is x_1=2. The functional margins t_i f(x_i), in the listed point order, are 2, 1, 1, 2. Thus (1,0) and (3,0) lie on the supporting planes. Since ||w||=1, the one-sided margin is 1 and the full width is 2.

No numerical correction is needed. The phrase "interior points" for (0,1) and (4,1) means points farther into their correctly classified half-spaces, outside the margin band.

## Q3. Slack and hinge loss

Both values are correct:

$$\xi_1=\max(0,1-0.4)=0.6,\qquad
\xi_2=\max(0,1-(-0.5))=1.5.$$

The second point is misclassified. The first is correctly classified but lies inside the margin, so its hinge loss is positive. These are the minimum feasible slack values for the given scores.

## Q4. Soft margin and C

The written objective and constraints are correct:

$$\min_{w,b,\xi}\frac12\lVert w\rVert^2+C\sum_i\xi_i,
\quad t_i(w^Tx_i+b)\geq1-\xi_i,\quad\xi_i\geq0.$$

**Wording correction:** a larger C penalizes margin violations more heavily relative to the norm penalty. It can encourage a narrower margin, but does not force a strictly narrower margin for every increase of C. It can also increase sensitivity to noise; overfitting and lower validation accuracy are possible, not inevitable.

The HAR experiment illustrates this distinction: C=0.01 gives validation Macro-F1 0.947470, while C=10 achieves training Macro-F1 1.0 but validation Macro-F1 0.936840. These results support the need for validation, not a universal rule about all datasets.

## Review outcome

No arithmetic errors were found. The main revisions clarify one-sided versus full margin width, the scope of the hard-margin support-vector definition, and conditional rather than absolute claims about increasing C.
