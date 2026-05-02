1. Code should be compact and readable
   - Ideally everything is small methods on a class, no loose functions
   - There should only be one class per file
   - Use inheritence and composition to keep individual classes small and focussed
   - Logically grouped code (especially blocks) should be separated by a blank newline

2. Code should be reusable
   - Don't keep writing the same kind of calculations, algorithms, or logic across many files
   - Instead, create modules, classes, and other reusable structures with named wrappers

3. Never use fallbacks
   - Never use silent fallbacks, just throw an exception when something isn't as it should be
   - Never hide issues, expose the flaws of the system so we can fix them

4. No optional or alternative paths, and no magic or tunable values
   - There is only one system, and that is the full system with everything wired in
   - Use dynamic/derived values in favor of tunable parameters