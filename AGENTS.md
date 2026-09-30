# Problem-solving instructions

For non-trivial tasks, do not immediately implement the first solution.

Use a Best-of-N approach:

1. Understand the problem and success criteria.
2. Generate 4–6 substantially different candidate solutions.
3. Compare them for:
   - correctness
   - compatibility
   - simplicity
   - maintainability
   - performance
   - regression risk
   - verifiability
4. Critically review each candidate.
5. Select the 2 strongest candidates.
6. Verify them using tests, compilation, runtime behavior, logs, or benchmarks whenever possible.
7. Choose the solution supported by the strongest evidence.
8. Implement only the selected solution.
9. Run tests after implementation.
10. If verification fails, analyze the failure and try another approach.

For trivial changes, skip Best-of-N and implement the obvious solution directly.

Prefer verified correctness over confidence or complexity.

Do not output lengthy internal reasoning. Only summarize:
- approaches considered
- selected approach
- changes made
- verification results
- remaining risks