// Checks for the web page's answer renderer (api/static/index.html), run by tests/test_page.py with Node.js.
// Usage: node page_render_checks.js <file exporting renderMarkdown and renderAnswer>
const assert = require("assert");
const { renderMarkdown, renderAnswer } = require(process.argv[2]);

const cite = (n) => `<a href="#src-${n}" class="cite" data-n="${n}" aria-label="source ${n}">[${n}]</a>`;
assert.strictEqual(renderMarkdown("ICANS in 40% [1]."), `<p>ICANS in 40% ${cite(1)}.</p>`);
assert.strictEqual(renderMarkdown("A [1][3] and [2, 4]."), `<p>A ${cite(1)}${cite(3)} and ${cite(2)}${cite(4)}.</p>`);
assert.strictEqual(renderMarkdown("median 26 [22-30]"), "<p>median 26 [22-30]</p>");  // statistics stay text
assert.strictEqual(renderMarkdown("**Bold**\n- one\n- two\n\nAfter."),
                   "<p><strong>Bold</strong></p><ul><li>one</li><li>two</li></ul><p>After.</p>");
assert.strictEqual(renderMarkdown("1. first\n2. second"), "<ol><li>first</li><li>second</li></ol>");
assert.strictEqual(renderMarkdown("## Heading\ntext"), "<h3>Heading</h3><p>text</p>");
// An answer can never add its own HTML to the page
assert.strictEqual(renderMarkdown('<img src=x onerror="alert(1)"> <script>bad()</script>'),
                   "<p>&lt;img src=x onerror=&quot;alert(1)&quot;&gt; &lt;script&gt;bad()&lt;/script&gt;</p>");
// Text after a "Details for clinicians and researchers" line is folded away
const folded = renderAnswer("Short [1].\n\n**Details for clinicians and researchers:**\nGrade 3 [4].");
assert(folded.includes('<details class="more">') && folded.indexOf("Grade 3") > folded.indexOf("<details"));
assert(!renderAnswer("No details here [1].").includes("<details"));
console.log("page renderer: all checks passed");
