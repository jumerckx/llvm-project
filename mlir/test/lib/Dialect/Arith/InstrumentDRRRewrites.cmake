file(READ ${INPUT} contents)
string(REPLACE "// Rewrite\n" "// Rewrite\nRewritePhaseScope rewritePhaseScope(rewriter);\n"
       contents "${contents}")
file(WRITE ${OUTPUT} "${contents}")
