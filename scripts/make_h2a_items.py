"""Build the H2a (second-token recovery) item set for the Qwen3 tokenizer: data/h2a_items.json.

Each item is a prompt whose workspace at the readout position (always the last prompt token) should carry a
multi-token surface `string`; H2a asks whether the true second piece of that string is the top-ranked extension of
the prompt-blind step readout W_U J(s1) h given the true first piece s1. Two kinds:
  answer  the greedy continuation of the prompt is `string` itself (capital, country, element, surname, animal,
          translation, number word, chemical formula, year, arithmetic result). Gate at run time: greedy-decode
          len(pieces) tokens and require an exact match.
  bridge  a two-hop prompt whose hidden intermediate is `string`; `target` (str or list of acceptable strings) is
          the task answer, used only for the correctness gate (the intermediate never appears in the prompt).

Sources (field `source`):
  lens_eval    anthropics/jacobian-lens data/evaluations/lens-eval-{multihop,multilingual,order-ops,association,
               poetry,typo}.json. Bridge items: every intermediate of every item; the readout is the token before
               `target`, i.e. the last prompt token because prompts end just before the target. Items with `target`
               also yield an answer item (string = target with the leading space the prompt implies). Items whose
               prompt is a chat list are skipped. category = file slug (+ "-target" for answer items).
  probe_swap   data/experiments/probe-swap.json: 90 two-hop prompts (bridge = `intermediate`, target = `answer`),
               plus an answer item for each multi-token `answer`. category = the file's `category`.
  flexgen      data/experiments/flexible-generalization.json: 4 categories x 4 args x 4 templates, single hop; answer
               items only. category = "{category}-{func}".
  curated      tables written below: countries (capital / currency / language / continent), famous people (work ->
               surname -> nationality / native language), chemical elements (symbol -> name -> atomic number),
               animals, translations (es / fr / de / it / pt), number words, chemical formulas, years and arithmetic
               (digit strings after a prompt ending in a space, as in Anthropic's order-ops prompts). Bridge templates
               hide a multi-token country behind its capital or (unique) currency, or a multi-token surname behind a
               description of the person's work.
Leading space: `string` carries the leading space it would have as a continuation (" Estonia"), except after a
prompt that ends in whitespace or an opening quote / bracket, where it has none ("1945" after "... the year ").
Filtering (in order): (1) pieces = tok.encode(string, add_special_tokens=False) must have >= 2 pieces; (2) the first
piece must contain a letter or digit (never a bare space or punctuation) - if " x" fails this, "x" without the
leading space is tried once (this is what rescues digit strings, " 12" -> [" ", "1", "2"] but "12" -> ["1", "2"]; a single-token " word" is
never retried as "word");
(3) tok.decode(pieces) must round-trip to `string`; (4) the string must not occur in the prompt (case-insensitive:
recall, not copying); (5) exact (prompt, string) duplicates are dropped, first source wins, in the order lens_eval,
probe_swap, flexgen, curated; (6) at most --cap-string (5) items per (source, category, string), so one intermediate
(" português" on nine multilingual prompts) cannot dominate; (7) each (source, category) is capped at --cap (60) items
by a seeded random sample so translations and country facts do not swamp the set. Facts are chosen to be uncontroversial; where a fact is
contested or the model's canonical answer is ambiguous the field is left None and no item is made from it.
Prompt formats were chosen after checking Qwen3-1.7B's greedy continuations on CPU: translations end in an open
double quote (the model otherwise mirrors the quotes, " 'mariposa'"), currency asks for "the unit of currency in X"
(otherwise the model prefixes the demonym, " Pakistani rupee"), number-word sums say "Twelve plus seven equals"
(the "written in words" preamble made the model answer in digits). With these, ~40% of all items pass the greedy
gate on Qwen3-1.7B; translations, years, arithmetic, capitals, elements and element-number bridges pass most often.
Usage: python scripts/make_h2a_items.py --anthropic /path/to/jacobian-lens/data [--out data/h2a_items.json] [--cap 60]
       [--cap-string 5]; prints counts per source / kind / category. Every item carries `p1_chars` (letters in the
first piece after stripping the space) so runs can drop one-letter first pieces (" Q" + "uito") if wanted.

Wave 2 (agent G). Two more item sets from the SAME templates and filters (the Anthropic source directory is not
available locally, so both are curated-only; --anthropic is honoured when present):
  --single    keeps only strings that are ONE Qwen token (the default builder drops them at filter (1)); writes
              data/single_items.json and data/single_split.json (seed 0, tuning / held-out halves of the answer ids,
              the h2a_split.json rule). "answer" gating for these items is: the greedy first token equals the token.
  --phrases   the WORD-PHRASE set (closer to WorkspaceBench's multi-token concepts): keeps only strings of >= 2
              pieces where EVERY piece is a whole space-prefixed word token (each piece decodes to " <letters/digits>";
              the first piece follows the prompt's leading-space rule, i.e. no space after an opening quote), from the
              curated templates plus the PHRASE_* tables below (ambiguous first words: New / South / North / San /
              Saint / United / West ...; people as first + last name; chemistry names; events; landmarks; US states;
              cities; films and characters) and bridge templates hiding a word-level entity ("... whose capital is
              Pretoria is" hides " South Africa"; "The nationality of the physicist who developed the theory of
              relativity was" hides " Albert Einstein"). Every item carries `shared_s1_group` = the first piece's word,
              so analyses can compare items that share the first word (does s1 fix s2?). Writes data/phrase_items.json
              and data/phrase_split.json.
"""
import argparse, glob, hashlib, json, os, random, sys

# ---------------------------------------------------------------- curated tables -------------------------------
# country, capital, currency, official / main language, continent. None = ambiguous, no item from that field.
COUNTRIES = [
    ("Kazakhstan", "Astana", "tenge", "Kazakh", "Asia"), ("Uzbekistan", "Tashkent", "som", "Uzbek", "Asia"),
    ("Kyrgyzstan", "Bishkek", "som", "Kyrgyz", "Asia"), ("Tajikistan", "Dushanbe", "somoni", "Tajik", "Asia"),
    ("Turkmenistan", "Ashgabat", "manat", "Turkmen", "Asia"), ("Azerbaijan", "Baku", "manat", "Azerbaijani", "Asia"),
    ("Armenia", "Yerevan", "dram", "Armenian", "Asia"), ("Mongolia", "Ulaanbaatar", "tugrik", "Mongolian", "Asia"),
    ("Bangladesh", "Dhaka", "taka", "Bengali", "Asia"), ("Nepal", "Kathmandu", "rupee", "Nepali", "Asia"),
    ("Bhutan", "Thimphu", "ngultrum", "Dzongkha", "Asia"), ("Maldives", "Malé", "rufiyaa", "Dhivehi", "Asia"),
    ("Sri Lanka", "Colombo", "rupee", None, "Asia"), ("Cambodia", "Phnom Penh", "riel", "Khmer", "Asia"),
    ("Laos", "Vientiane", "kip", "Lao", "Asia"), ("Myanmar", "Naypyidaw", "kyat", "Burmese", "Asia"),
    ("Vietnam", "Hanoi", "dong", "Vietnamese", "Asia"), ("Thailand", "Bangkok", "baht", "Thai", "Asia"),
    ("Malaysia", "Kuala Lumpur", "ringgit", "Malay", "Asia"), ("Indonesia", "Jakarta", "rupiah", "Indonesian", "Asia"),
    ("Philippines", "Manila", "peso", "Filipino", "Asia"), ("Afghanistan", "Kabul", "afghani", None, "Asia"),
    ("Pakistan", "Islamabad", "rupee", "Urdu", "Asia"), ("Iran", "Tehran", "rial", "Persian", "Asia"),
    ("Iraq", "Baghdad", "dinar", "Arabic", "Asia"), ("Saudi Arabia", "Riyadh", "riyal", "Arabic", "Asia"),
    ("Oman", "Muscat", "rial", "Arabic", "Asia"), ("Qatar", "Doha", "riyal", "Arabic", "Asia"),
    ("Bahrain", "Manama", "dinar", "Arabic", "Asia"), ("Jordan", "Amman", "dinar", "Arabic", "Asia"),
    ("Lebanon", "Beirut", "pound", "Arabic", "Asia"), ("Syria", "Damascus", "pound", "Arabic", "Asia"),
    ("Turkey", "Ankara", "lira", "Turkish", None), ("South Korea", "Seoul", "won", "Korean", "Asia"),
    ("North Korea", "Pyongyang", "won", "Korean", "Asia"), ("Japan", "Tokyo", "yen", "Japanese", "Asia"),
    ("China", "Beijing", "yuan", ["Mandarin", "Chinese"], "Asia"), ("India", "New Delhi", "rupee", ["Hindi"], "Asia"),
    ("Ukraine", "Kyiv", "hryvnia", "Ukrainian", "Europe"), ("Belarus", "Minsk", "ruble", None, "Europe"),
    ("Lithuania", "Vilnius", "euro", "Lithuanian", "Europe"), ("Latvia", "Riga", "euro", "Latvian", "Europe"),
    ("Estonia", "Tallinn", "euro", "Estonian", "Europe"), ("Moldova", "Chisinau", "leu", "Romanian", "Europe"),
    ("Romania", "Bucharest", "leu", "Romanian", "Europe"), ("Bulgaria", "Sofia", None, "Bulgarian", "Europe"),
    ("Serbia", "Belgrade", "dinar", "Serbian", "Europe"), ("Croatia", "Zagreb", "euro", "Croatian", "Europe"),
    ("Slovenia", "Ljubljana", "euro", ["Slovene", "Slovenian"], "Europe"), ("Bosnia and Herzegovina", "Sarajevo", None, None, "Europe"),
    ("North Macedonia", "Skopje", "denar", "Macedonian", "Europe"), ("Albania", "Tirana", "lek", "Albanian", "Europe"),
    ("Montenegro", "Podgorica", "euro", None, "Europe"), ("Slovakia", "Bratislava", "euro", "Slovak", "Europe"),
    ("Czech Republic", "Prague", "koruna", "Czech", "Europe"), ("Hungary", "Budapest", "forint", "Hungarian", "Europe"),
    ("Poland", "Warsaw", "zloty", "Polish", "Europe"), ("Austria", "Vienna", "euro", "German", "Europe"),
    ("Switzerland", "Bern", "franc", None, "Europe"), ("Netherlands", "Amsterdam", "euro", "Dutch", "Europe"),
    ("Belgium", "Brussels", "euro", None, "Europe"), ("Denmark", "Copenhagen", "krone", "Danish", "Europe"),
    ("Norway", "Oslo", "krone", "Norwegian", "Europe"), ("Sweden", "Stockholm", "krona", "Swedish", "Europe"),
    ("Finland", "Helsinki", "euro", "Finnish", "Europe"), ("Iceland", "Reykjavik", "krona", "Icelandic", "Europe"),
    ("Ireland", "Dublin", "euro", None, "Europe"), ("Portugal", "Lisbon", "euro", "Portuguese", "Europe"),
    ("Spain", "Madrid", "euro", "Spanish", "Europe"), ("Greece", "Athens", "euro", "Greek", "Europe"),
    ("Malta", "Valletta", "euro", "Maltese", "Europe"), ("Germany", "Berlin", "euro", "German", "Europe"),
    ("Morocco", "Rabat", "dirham", "Arabic", "Africa"), ("Algeria", "Algiers", "dinar", "Arabic", "Africa"),
    ("Tunisia", "Tunis", "dinar", "Arabic", "Africa"), ("Libya", "Tripoli", "dinar", "Arabic", "Africa"),
    ("Egypt", "Cairo", "pound", "Arabic", "Africa"), ("Sudan", "Khartoum", "pound", "Arabic", "Africa"),
    ("Ethiopia", "Addis Ababa", "birr", "Amharic", "Africa"), ("Eritrea", "Asmara", "nakfa", None, "Africa"),
    ("Somalia", "Mogadishu", "shilling", "Somali", "Africa"), ("Kenya", "Nairobi", "shilling", ["Swahili", "English"], "Africa"),
    ("Uganda", "Kampala", "shilling", ["English", "Swahili"], "Africa"), ("Tanzania", "Dodoma", "shilling", "Swahili", "Africa"),
    ("Rwanda", "Kigali", "franc", "Kinyarwanda", "Africa"), ("Zambia", "Lusaka", "kwacha", "English", "Africa"),
    ("Malawi", "Lilongwe", "kwacha", ["English", "Chichewa"], "Africa"), ("Zimbabwe", "Harare", None, "English", "Africa"),
    ("Mozambique", "Maputo", "metical", "Portuguese", "Africa"), ("Angola", "Luanda", "kwanza", "Portuguese", "Africa"),
    ("Namibia", "Windhoek", "dollar", "English", "Africa"), ("Botswana", "Gaborone", "pula", ["English", "Setswana"], "Africa"),
    ("South Africa", None, "rand", None, "Africa"), ("Madagascar", "Antananarivo", "ariary", "Malagasy", "Africa"),
    ("Ghana", "Accra", "cedi", "English", "Africa"), ("Nigeria", "Abuja", "naira", "English", "Africa"),
    ("Senegal", "Dakar", "franc", "French", "Africa"), ("Mali", "Bamako", "franc", "French", "Africa"),
    ("Niger", "Niamey", "franc", "French", "Africa"), ("Burkina Faso", "Ouagadougou", "franc", "French", "Africa"),
    ("Liberia", "Monrovia", "dollar", "English", "Africa"), ("Sierra Leone", "Freetown", "leone", "English", "Africa"),
    ("Guinea", "Conakry", "franc", "French", "Africa"), ("Cameroon", "Yaoundé", "franc", ["French", "English"], "Africa"),
    ("Gabon", "Libreville", "franc", "French", "Africa"), ("Canada", "Ottawa", "dollar", None, "North America"),
    ("Mexico", "Mexico City", "peso", "Spanish", "North America"), ("Guatemala", "Guatemala City", "quetzal", "Spanish", "North America"),
    ("Honduras", "Tegucigalpa", "lempira", "Spanish", "North America"), ("Nicaragua", "Managua", "córdoba", "Spanish", "North America"),
    ("Costa Rica", "San José", "colón", "Spanish", "North America"), ("Panama", "Panama City", "balboa", "Spanish", "North America"),
    ("El Salvador", "San Salvador", None, "Spanish", "North America"), ("Cuba", "Havana", "peso", "Spanish", "North America"),
    ("Jamaica", "Kingston", "dollar", "English", "North America"), ("Haiti", "Port-au-Prince", "gourde", ["French", "Haitian"], "North America"),
    ("Dominican Republic", "Santo Domingo", "peso", "Spanish", "North America"), ("Bahamas", "Nassau", "dollar", "English", "North America"),
    ("Colombia", "Bogotá", "peso", "Spanish", "South America"), ("Venezuela", "Caracas", "bolívar", "Spanish", "South America"),
    ("Ecuador", "Quito", "dollar", "Spanish", "South America"), ("Peru", "Lima", "sol", "Spanish", "South America"),
    ("Bolivia", None, "boliviano", "Spanish", "South America"), ("Chile", "Santiago", "peso", "Spanish", "South America"),
    ("Argentina", "Buenos Aires", "peso", "Spanish", "South America"), ("Uruguay", "Montevideo", "peso", "Spanish", "South America"),
    ("Paraguay", "Asunción", "guarani", ["Spanish", "Guarani"], "South America"), ("Brazil", "Brasília", "real", "Portuguese", "South America"),
    ("Guyana", "Georgetown", "dollar", "English", "South America"), ("Suriname", "Paramaribo", "dollar", "Dutch", "South America"),
    ("Australia", "Canberra", "dollar", "English", ["Oceania", "Australia"]), ("New Zealand", "Wellington", "dollar", "English", "Oceania"),
    ("Papua New Guinea", "Port Moresby", "kina", None, "Oceania"), ("Fiji", "Suva", "dollar", None, "Oceania"),
]

# description of the work (fills "the ... of ..."), surname, nationality, native language
PEOPLE = [
    ("the composer of the ballet Swan Lake", "Tchaikovsky", "Russian", "Russian"),
    ("the author of Crime and Punishment", "Dostoevsky", "Russian", "Russian"),
    ("the author of War and Peace", "Tolstoy", "Russian", "Russian"),
    ("the author of The Cherry Orchard", "Chekhov", "Russian", "Russian"),
    ("the author of Doctor Zhivago", "Pasternak", "Russian", "Russian"),
    ("the author of The Gulag Archipelago", "Solzhenitsyn", "Russian", "Russian"),
    ("the author of Fathers and Sons", "Turgenev", "Russian", "Russian"),
    ("the poet who wrote Eugene Onegin", "Pushkin", "Russian", "Russian"),
    ("the poet who wrote A Hero of Our Time", "Lermontov", "Russian", "Russian"),
    ("the composer of The Rite of Spring", "Stravinsky", "Russian", "Russian"),
    ("the composer of Pictures at an Exhibition", "Mussorgsky", "Russian", "Russian"),
    ("the composer of Scheherazade", "Rimsky-Korsakov", "Russian", "Russian"),
    ("the composer of the Leningrad Symphony", "Shostakovich", "Russian", "Russian"),
    ("the composer of Peter and the Wolf", "Prokofiev", "Russian", "Russian"),
    ("the composer of Rhapsody on a Theme of Paganini", "Rachmaninoff", "Russian", "Russian"),
    ("the chemist who created the periodic table", "Mendeleev", "Russian", "Russian"),
    ("the composer of the Sabre Dance", "Khachaturian", "Armenian", "Armenian"),
    ("the painter of The Starry Night", "Van Gogh", "Dutch", "Dutch"),
    ("the painter of The Night Watch", "Rembrandt", "Dutch", "Dutch"),
    ("the painter of Girl with a Pearl Earring", "Vermeer", "Dutch", "Dutch"),
    ("the physicist who formulated the uncertainty principle", "Heisenberg", "German", "German"),
    ("the inventor of the movable-type printing press in Europe", "Gutenberg", "German", "German"),
    ("the philosopher who wrote Thus Spoke Zarathustra", "Nietzsche", "German", "German"),
    ("the author of Faust", "Goethe", "German", "German"),
    ("the physicist who discovered X-rays", "Röntgen", "German", "German"),
    ("the philosopher who wrote the Critique of Pure Reason", "Kant", "German", "German"),
    ("the philosopher who wrote Being and Time", "Heidegger", "German", "German"),
    ("the mathematician who introduced the Riemann hypothesis", "Riemann", "German", "German"),
    ("the mathematician after whom the normal distribution is named", "Gauss", "German", "German"),
    ("the astronomer who discovered the laws of planetary motion", "Kepler", "German", "German"),
    ("the composer of the Ring cycle", "Wagner", "German", "German"),
    ("the composer of Ode to Joy", "Beethoven", "German", "German"),
    ("the composer of Carmina Burana", "Orff", "German", "German"),
    ("the author of The Tin Drum", "Grass", "German", "German"),
    ("the physicist whose wave equation governs quantum mechanics", "Schrödinger", "Austrian", "German"),
    ("the composer of The Magic Flute", "Mozart", "Austrian", "German"),
    ("the philosopher who wrote the Tractatus Logico-Philosophicus", "Wittgenstein", "Austrian", "German"),
    ("the composer of the Blue Danube waltz", "Strauss", "Austrian", "German"),
    ("the psychoanalyst who wrote The Interpretation of Dreams", "Freud", "Austrian", "German"),
    ("the painter of The Kiss (1908)", "Klimt", "Austrian", "German"),
    ("the composer of the Unfinished Symphony", "Schubert", "Austrian", "German"),
    ("the astronomer who placed the Sun at the centre of the solar system in 1543", "Copernicus", "Polish", "Polish"),
    ("the composer of the Nocturnes and Polonaises for piano", "Chopin", "Polish", "Polish"),
    ("the painter of the Sistine Chapel ceiling", "Michelangelo", "Italian", "Italian"),
    ("the painter of The Birth of Venus", "Botticelli", "Italian", "Italian"),
    ("the author of The Prince", "Machiavelli", "Italian", "Italian"),
    ("the composer of Madama Butterfly", "Puccini", "Italian", "Italian"),
    ("the composer of The Four Seasons", "Vivaldi", "Italian", "Italian"),
    ("the composer of the opera Aida", "Verdi", "Italian", "Italian"),
    ("the composer of The Barber of Seville", "Rossini", "Italian", "Italian"),
    ("the author of The Name of the Rose", "Eco", "Italian", "Italian"),
    ("the author of Don Quixote", "Cervantes", "Spanish", "Spanish"),
    ("the painter of Guernica", "Picasso", "Spanish", "Spanish"),
    ("the painter of The Persistence of Memory", "Dalí", "Spanish", "Spanish"),
    ("the architect of the Sagrada Família", "Gaudí", "Spanish", "Spanish"),
    ("the conquistador who conquered the Inca Empire", "Pizarro", "Spanish", "Spanish"),
    ("the conquistador who conquered the Aztec Empire", "Cortés", "Spanish", "Spanish"),
    ("the explorer whose expedition first circumnavigated the globe", "Magellan", "Portuguese", "Portuguese"),
    ("the author of The Alchemist", "Coelho", "Brazilian", "Portuguese"),
    ("the author of Madame Bovary", "Flaubert", "French", "French"),
    ("the composer of the opera Carmen", "Bizet", "French", "French"),
    ("the author of The Stranger", "Camus", "French", "French"),
    ("the author of In Search of Lost Time", "Proust", "French", "French"),
    ("the painter of Impression, Sunrise", "Monet", "French", "French"),
    ("the composer of Boléro", "Ravel", "French", "French"),
    ("the composer of Clair de Lune", "Debussy", "French", "French"),
    ("the sculptor of The Thinker", "Rodin", "French", "French"),
    ("the philosopher who wrote Discourse on the Method", "Descartes", "French", "French"),
    ("the author of Candide", "Voltaire", "French", "French"),
    ("the microbiologist who developed pasteurization", "Pasteur", "French", "French"),
    ("the composer of the Symphonie fantastique", "Berlioz", "French", "French"),
    ("the composer of Peer Gynt", "Grieg", "Norwegian", "Norwegian"),
    ("the playwright who wrote Hedda Gabler", "Ibsen", "Norwegian", "Norwegian"),
    ("the painter of The Scream", "Munch", "Norwegian", "Norwegian"),
    ("the composer of Finlandia", "Sibelius", "Finnish", "Finnish"),
    ("the composer of the New World Symphony", "Dvořák", "Czech", "Czech"),
    ("the composer of Má vlast (The Moldau)", "Smetana", "Czech", "Czech"),
    ("the composer of the Hungarian Rhapsodies", "Liszt", "Hungarian", "Hungarian"),
    ("the philosopher who wrote Either/Or", "Kierkegaard", "Danish", "Danish"),
    ("the author of The Ugly Duckling", "Andersen", "Danish", "Danish"),
    ("the physicist who proposed the 1913 model of the atom with electron orbits", "Bohr", "Danish", "Danish"),
    ("the astronomer who observed the stars from Uraniborg", "Brahe", "Danish", "Danish"),
    ("the playwright who wrote Miss Julie", "Strindberg", "Swedish", "Swedish"),
    ("the chemist who invented dynamite", "Nobel", "Swedish", "Swedish"),
    ("the botanist who created binomial nomenclature", "Linnaeus", "Swedish", "Swedish"),
    ("the author of Pride and Prejudice", "Austen", ["English", "British"], "English"),
    ("the author of Great Expectations", "Dickens", ["English", "British"], "English"),
    ("the author of Frankenstein", "Shelley", ["English", "British"], "English"),
    ("the naturalist who wrote On the Origin of Species", "Darwin", ["English", "British"], "English"),
    ("the author of Brave New World", "Huxley", ["English", "British"], "English"),
    ("the author of The Lord of the Rings", "Tolkien", ["English", "British"], "English"),
    ("the author of A Brief History of Time", "Hawking", ["English", "British"], "English"),
    ("the physicist who discovered the neutron", "Chadwick", ["English", "British"], "English"),
    ("the philosopher who wrote Leviathan", "Hobbes", ["English", "British"], "English"),
    ("the physician who developed the smallpox vaccine", "Jenner", ["English", "British"], "English"),
    ("the composer of The Planets", "Holst", ["English", "British"], "English"),
    ("the composer of the Enigma Variations", "Elgar", ["English", "British"], "English"),
    ("the mathematician who proved Fermat's Last Theorem", "Wiles", ["English", "British"], "English"),
    ("the physicist who unified electricity and magnetism in four equations", "Maxwell", ["Scottish", "British"], "English"),
    ("the mathematician who solved the Seven Bridges of Königsberg problem", "Euler", "Swiss", "German"),
    ("the author of The Great Gatsby", "Fitzgerald", "American", "English"),
    ("the author of The Old Man and the Sea", "Hemingway", "American", "English"),
    ("the author of Moby-Dick", "Melville", "American", "English"),
    ("the author of The Grapes of Wrath", "Steinbeck", "American", "English"),
    ("the author of The Sound and the Fury", "Faulkner", "American", "English"),
    ("the composer of Rhapsody in Blue", "Gershwin", "American", "English"),
    ("the author of Things Fall Apart", "Achebe", "Nigerian", None),
    ("the poet who wrote Gitanjali", "Tagore", "Indian", "Bengali"),
    ("the director of Seven Samurai", "Kurosawa", "Japanese", "Japanese"),
    ("the author of Norwegian Wood", "Murakami", "Japanese", "Japanese"),
    ("the author of Snow Country", "Kawabata", "Japanese", "Japanese"),
    ("the author of Pedro Páramo", "Rulfo", "Mexican", "Spanish"),
    ("the painter of The Two Fridas", "Kahlo", "Mexican", "Spanish"),
    ("the author of The Death of Artemio Cruz", "Fuentes", "Mexican", "Spanish"),
    ("the poet who wrote Twenty Love Poems and a Song of Despair", "Neruda", "Chilean", "Spanish"),
    ("the author of The House of the Spirits", "Allende", "Chilean", "Spanish"),
    ("the author of Ficciones", "Borges", ["Argentine", "Argentinian"], "Spanish"),
    ("the author of Hopscotch (Rayuela)", "Cortázar", ["Argentine", "Argentinian"], "Spanish"),
    ("the author of Conversation in the Cathedral", "Vargas Llosa", "Peruvian", "Spanish"),
    ("the author of One Hundred Years of Solitude", "García Márquez", "Colombian", "Spanish"),
    ("the philosopher who taught Alexander the Great", "Aristotle", "Greek", "Greek"),
    ("the philosopher who wrote The Republic", "Plato", "Greek", "Greek"),
    ("the mathematician who wrote the Elements", "Euclid", "Greek", "Greek"),
    ("the mathematician who discovered the law of buoyancy", "Archimedes", "Greek", "Greek"),
    ("the mathematician after whom the right-triangle theorem is named", "Pythagoras", "Greek", "Greek"),
]

# symbol, name, atomic number
ELEMENTS = [
    ("Li", "lithium", 3), ("Be", "beryllium", 4), ("Na", "sodium", 11), ("Mg", "magnesium", 12), ("Al", "aluminium", 13),
    ("Si", "silicon", 14), ("P", "phosphorus", 15), ("S", "sulfur", 16), ("Cl", "chlorine", 17), ("Ar", "argon", 18),
    ("K", "potassium", 19), ("Ca", "calcium", 20), ("Sc", "scandium", 21), ("Ti", "titanium", 22), ("V", "vanadium", 23),
    ("Cr", "chromium", 24), ("Mn", "manganese", 25), ("Fe", "iron", 26), ("Co", "cobalt", 27), ("Ni", "nickel", 28),
    ("Cu", "copper", 29), ("Zn", "zinc", 30), ("Ga", "gallium", 31), ("Ge", "germanium", 32), ("As", "arsenic", 33),
    ("Se", "selenium", 34), ("Br", "bromine", 35), ("Kr", "krypton", 36), ("Rb", "rubidium", 37), ("Sr", "strontium", 38),
    ("Y", "yttrium", 39), ("Zr", "zirconium", 40), ("Nb", "niobium", 41), ("Mo", "molybdenum", 42), ("Tc", "technetium", 43),
    ("Ru", "ruthenium", 44), ("Rh", "rhodium", 45), ("Pd", "palladium", 46), ("Ag", "silver", 47), ("Cd", "cadmium", 48),
    ("In", "indium", 49), ("Sn", "tin", 50), ("Sb", "antimony", 51), ("Te", "tellurium", 52), ("I", "iodine", 53),
    ("Xe", "xenon", 54), ("Cs", "caesium", 55), ("Ba", "barium", 56), ("La", "lanthanum", 57), ("Ce", "cerium", 58),
    ("Nd", "neodymium", 60), ("Sm", "samarium", 62), ("Eu", "europium", 63), ("Gd", "gadolinium", 64), ("Tb", "terbium", 65),
    ("Dy", "dysprosium", 66), ("Ho", "holmium", 67), ("Er", "erbium", 68), ("Yb", "ytterbium", 70), ("Lu", "lutetium", 71),
    ("Hf", "hafnium", 72), ("Ta", "tantalum", 73), ("W", "tungsten", 74), ("Re", "rhenium", 75), ("Os", "osmium", 76),
    ("Ir", "iridium", 77), ("Pt", "platinum", 78), ("Au", "gold", 79), ("Hg", "mercury", 80), ("Tl", "thallium", 81),
    ("Pb", "lead", 82), ("Bi", "bismuth", 83), ("Po", "polonium", 84), ("Rn", "radon", 86), ("Fr", "francium", 87),
    ("Ra", "radium", 88), ("Th", "thorium", 90), ("U", "uranium", 92), ("Np", "neptunium", 93), ("Pu", "plutonium", 94),
    ("Am", "americium", 95), ("Cm", "curium", 96), ("Es", "einsteinium", 99),
]

# description -> animal (the greedy answer expected)
ANIMALS = [
    ("The animal with the longest neck is the", "giraffe"), ("The fastest land animal is the", "cheetah"),
    ("The largest living bird, which cannot fly, is the", "ostrich"), ("The Australian marsupial that hops and carries its young in a pouch is the", "kangaroo"),
    ("The large African mammal with one or two horns on its snout is the", "rhinoceros"), ("The large semi-aquatic African mammal with a huge mouth is the", "hippopotamus"),
    ("The great ape that shares the most DNA with humans is the", "chimpanzee"), ("The red-haired great ape of Borneo and Sumatra is the", "orangutan"),
    ("The egg-laying mammal with a duck-like bill is the", "platypus"), ("The armoured mammal that rolls into a ball is the", "armadillo"),
    ("The lizard that changes colour is the", "chameleon"), ("The eight-legged creature that spins webs is the", "spider"),
    ("The pink wading bird that stands on one leg is the", "flamingo"), ("The large seabird with the widest wingspan is the", "albatross"),
    ("The tallest flying bird, known for its long legs and dancing, is the", "crane"), ("The Australian marsupial that eats eucalyptus leaves is the", "koala"),
    ("The spotted big cat of the Americas is the", "jaguar"), ("The laughing scavenger of the African savanna is the", "hyena"),
    ("The largest reptile alive today is the saltwater", "crocodile"), ("The eight-armed marine animal that squirts ink is the", "octopus"),
    ("The marine mammal with two long tusks that lives in the Arctic is the", "walrus"), ("The Arctic whale with a single long spiral tusk is the", "narwhal"),
    ("The gentle sea mammal sometimes called a sea cow is the", "manatee"), ("The mammal covered in sharp quills is the", "porcupine"),
    ("The small spiny mammal that rolls into a ball is the", "hedgehog"), ("The masked mammal that washes its food is the", "raccoon"),
    ("The small mongoose of the Kalahari that stands upright to keep watch is the", "meerkat"), ("The primate found only on Madagascar is the", "lemur"),
    ("The largest living primate is the", "gorilla"), ("The largest rodent in the world is the", "capybara"),
    ("The scaly mammal that eats ants and rolls into a ball is the", "pangolin"), ("The striped horse-like animal of Africa is the", "zebra"),
    ("The bird with a huge colourful beak that lives in Central and South America is the", "toucan"), ("The smallest bird, which can hover and fly backwards, is the", "hummingbird"),
    ("The flightless bird of Antarctica is the", "penguin"), ("The venomous snake with a hood is the", "cobra"),
    ("The largest snake by weight, found in South America, is the", "anaconda"), ("The large flightless bird of Australia, second in size only to the ostrich, is the", "emu"),
    ("The bird famous for its brightly coloured tail feathers is the", "peacock"), ("The black-and-white bear native to China is the giant", "panda"),
    ("The largest species of deer, with broad antlers, is the", "moose"), ("The bird of prey that is the national bird of the United States is the bald", "eagle"),
    ("The small striped rodent that stuffs its cheeks with food is the", "chipmunk"), ("The slow-moving Central American mammal that hangs upside down from trees is the", "sloth"),
    ("The venomous spider with a red hourglass on its abdomen is the black", "widow"), ("The amphibian that begins life as a tadpole is the", "frog"),
    ("The nocturnal bird that hoots is the", "owl"), ("The largest species of shark, which feeds on plankton, is the whale", "shark"),
    ("The wild dog of Australia is the", "dingo"), ("The lizard of Indonesia that is the largest living lizard is the Komodo", "dragon"),
]

# English -> (es, fr, de, it, pt); None where the usual translation is ambiguous or identical to English
TRANSLATIONS = [
    ("butterfly", "mariposa", "papillon", "Schmetterling", "farfalla", "borboleta"), ("library", "biblioteca", "bibliothèque", "Bibliothek", "biblioteca", "biblioteca"),
    ("strawberry", "fresa", "fraise", "Erdbeere", "fragola", "morango"), ("window", "ventana", "fenêtre", "Fenster", "finestra", "janela"),
    ("kitchen", "cocina", "cuisine", "Küche", "cucina", "cozinha"), ("Wednesday", "miércoles", "mercredi", "Mittwoch", "mercoledì", "quarta-feira"),
    ("yellow", "amarillo", "jaune", "gelb", "giallo", "amarelo"), ("cheese", "queso", "fromage", "Käse", "formaggio", "queijo"),
    ("apple", "manzana", "pomme", "Apfel", "mela", "maçã"), ("bread", "pan", "pain", "Brot", "pane", "pão"),
    ("water", "agua", "eau", "Wasser", "acqua", "água"), ("dog", "perro", "chien", "Hund", "cane", None),
    ("cat", "gato", "chat", "Katze", "gatto", "gato"), ("house", "casa", "maison", "Haus", "casa", "casa"),
    ("school", "escuela", "école", "Schule", "scuola", "escola"), ("airport", "aeropuerto", "aéroport", "Flughafen", "aeroporto", "aeroporto"),
    ("breakfast", "desayuno", "petit déjeuner", "Frühstück", "colazione", None), ("Thursday", "jueves", "jeudi", "Donnerstag", "giovedì", "quinta-feira"),
    ("Saturday", "sábado", "samedi", "Samstag", "sabato", "sábado"), ("Sunday", "domingo", "dimanche", "Sonntag", "domenica", "domingo"),
    ("January", "enero", "janvier", "Januar", "gennaio", "janeiro"), ("February", "febrero", "février", "Februar", "febbraio", "fevereiro"),
    ("tomorrow", "mañana", "demain", "morgen", "domani", "amanhã"), ("yesterday", "ayer", "hier", "gestern", "ieri", "ontem"),
    ("always", "siempre", "toujours", "immer", "sempre", "sempre"), ("never", "nunca", "jamais", "nie", "mai", "nunca"),
    ("umbrella", "paraguas", "parapluie", "Regenschirm", "ombrello", "guarda-chuva"), ("shoe", "zapato", "chaussure", "Schuh", "scarpa", "sapato"),
    ("chair", "silla", "chaise", "Stuhl", "sedia", "cadeira"), ("bed", "cama", "lit", "Bett", "letto", "cama"),
    ("bird", "pájaro", "oiseau", "Vogel", "uccello", "pássaro"), ("horse", "caballo", "cheval", "Pferd", "cavallo", "cavalo"),
    ("cow", "vaca", "vache", "Kuh", "mucca", "vaca"), ("chicken", "pollo", "poulet", "Huhn", "pollo", "frango"),
    ("pig", "cerdo", "cochon", "Schwein", "maiale", "porco"), ("rabbit", "conejo", "lapin", "Kaninchen", "coniglio", "coelho"),
    ("turtle", "tortuga", "tortue", "Schildkröte", "tartaruga", "tartaruga"), ("butter", "mantequilla", "beurre", None, "burro", "manteiga"),
    ("milk", "leche", "lait", "Milch", "latte", "leite"), ("sugar", "azúcar", "sucre", "Zucker", "zucchero", "açúcar"),
    ("salt", "sal", "sel", "Salz", "sale", "sal"), ("egg", "huevo", "œuf", "Ei", "uovo", "ovo"),
    ("rice", "arroz", "riz", "Reis", "riso", "arroz"), ("beach", "playa", "plage", "Strand", "spiaggia", "praia"),
    ("mountain", "montaña", "montagne", "Berg", "montagna", "montanha"), ("city", "ciudad", "ville", "Stadt", "città", "cidade"),
    ("world", "mundo", "monde", "Welt", "mondo", "mundo"), ("sun", "sol", "soleil", "Sonne", "sole", "sol"),
    ("moon", "luna", "lune", "Mond", "luna", "lua"), ("star", "estrella", "étoile", "Stern", "stella", "estrela"),
    ("sky", "cielo", "ciel", "Himmel", "cielo", "céu"), ("rain", "lluvia", "pluie", "Regen", "pioggia", "chuva"),
    ("snow", "nieve", "neige", "Schnee", "neve", "neve"), ("wind", "viento", "vent", None, "vento", "vento"),
    ("fire", "fuego", "feu", "Feuer", "fuoco", "fogo"), ("tree", "árbol", "arbre", "Baum", "albero", "árvore"),
    ("flower", "flor", "fleur", "Blume", "fiore", "flor"), ("forest", "bosque", "forêt", "Wald", "foresta", "floresta"),
    ("garden", "jardín", "jardin", "Garten", "giardino", "jardim"), ("key", "llave", "clé", "Schlüssel", "chiave", "chave"),
    ("door", "puerta", "porte", "Tür", "porta", "porta"), ("wall", "pared", "mur", "Wand", "muro", "parede"),
    ("hospital", None, "hôpital", "Krankenhaus", "ospedale", None), ("glove", "guante", "gant", "Handschuh", "guanto", "luva"),
    ("refrigerator", "refrigerador", "réfrigérateur", "Kühlschrank", "frigorifero", "geladeira"), ("birthday", "cumpleaños", "anniversaire", "Geburtstag", "compleanno", "aniversário"),
    ("Christmas", "Navidad", "Noël", "Weihnachten", "Natale", "Natal"), ("newspaper", "periódico", "journal", "Zeitung", "giornale", "jornal"),
    ("science", "ciencia", "science", "Wissenschaft", "scienza", "ciência"), ("freedom", "libertad", "liberté", "Freiheit", "libertà", "liberdade"),
    ("friendship", "amistad", "amitié", "Freundschaft", "amicizia", "amizade"), ("health", "salud", "santé", "Gesundheit", "salute", "saúde"),
    ("knowledge", "conocimiento", "connaissance", "Wissen", "conoscenza", "conhecimento"), ("question", "pregunta", "question", "Frage", "domanda", "pergunta"),
    ("answer", "respuesta", "réponse", "Antwort", "risposta", "resposta"), ("thank you", "gracias", "merci", "danke", "grazie", "obrigado"),
    ("hello", "hola", "bonjour", "hallo", "ciao", "olá"), ("goodbye", "adiós", "au revoir", "auf Wiedersehen", "arrivederci", "adeus"),
    ("please", "por favor", "s'il vous plaît", "bitte", "per favore", "por favor"), ("hundred", "cien", "cent", "hundert", "cento", "cem"),
    ("thousand", "mil", "mille", "tausend", "mille", "mil"), ("week", "semana", "semaine", "Woche", "settimana", "semana"),
    ("year", "año", "année", "Jahr", "anno", "ano"), ("morning", "mañana", "matin", "Morgen", "mattina", "manhã"),
    ("night", "noche", "nuit", "Nacht", "notte", "noite"), ("summer", "verano", "été", "Sommer", "estate", "verão"),
    ("winter", "invierno", "hiver", "Winter", "inverno", "inverno"), ("spring", "primavera", "printemps", "Frühling", "primavera", "primavera"),
    ("autumn", "otoño", "automne", "Herbst", "autunno", "outono"), ("red", "rojo", "rouge", "rot", "rosso", "vermelho"),
    ("green", "verde", "vert", "grün", "verde", "verde"), ("black", "negro", "noir", "schwarz", "nero", "preto"),
    ("white", "blanco", "blanc", "weiß", "bianco", "branco"), ("blue", "azul", "bleu", "blau", "blu", "azul"),
]
LANGS = [("Spanish", 1), ("French", 2), ("German", 3), ("Italian", 4), ("Portuguese", 5)]

NUMWORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
            12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
            20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def numword(n):
    if n in NUMWORDS: return NUMWORDS[n]
    if n < 100: return NUMWORDS[n - n % 10] + "-" + NUMWORDS[n % 10]
    return NUMWORDS[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + numword(n % 100))


# (a, b, op) with op in {+, x}: "a plus b" / "a times b" written in words -> result in words
NUMWORD_PROBLEMS = [(12, 7, "+"), (14, 5, "+"), (9, 8, "+"), (11, 6, "+"), (13, 13, "+"), (20, 15, "+"), (30, 12, "+"), (25, 25, "+"),
                    (40, 19, "+"), (50, 27, "+"), (60, 33, "+"), (70, 18, "+"), (80, 16, "+"), (45, 45, "+"), (33, 44, "+"), (6, 7, "+"),
                    (3, 6, "x"), (7, 3, "x"), (4, 4, "x"), (5, 5, "x"), (6, 6, "x"), (7, 7, "x"), (8, 8, "x"), (9, 9, "x"), (3, 9, "x"),
                    (4, 8, "x"), (6, 9, "x"), (7, 8, "x"), (5, 9, "x"), (4, 9, "x"), (11, 11, "x"), (12, 12, "x"), (10, 10, "x"), (2, 9, "x"),
                    (9, 11, "x"), (8, 11, "x"), (7, 11, "x"), (12, 5, "x"), (15, 3, "x"), (25, 4, "x")]

FORMULAS = [("water", "H2O"), ("carbon dioxide", "CO2"), ("table salt (sodium chloride)", "NaCl"), ("methane", "CH4"), ("ammonia", "NH3"),
            ("glucose", "C6H12O6"), ("sulfuric acid", "H2SO4"), ("hydrogen peroxide", "H2O2"), ("ethanol", "C2H5OH"), ("baking soda (sodium bicarbonate)", "NaHCO3"),
            ("calcium carbonate", "CaCO3"), ("nitrous oxide", "N2O"), ("sulfur dioxide", "SO2"), ("hydrochloric acid", "HCl"), ("sodium hydroxide", "NaOH"),
            ("potassium chloride", "KCl"), ("carbon monoxide", "CO"), ("nitric acid", "HNO3"), ("magnesium oxide", "MgO"), ("silicon dioxide", "SiO2"),
            ("iron(III) oxide (rust)", "Fe2O3"), ("calcium oxide (quicklime)", "CaO"), ("acetic acid", "CH3COOH"), ("propane", "C3H8"), ("ozone", "O3")]

YEARS = [("World War II ended", 1945), ("World War I began", 1914), ("the Berlin Wall fell", 1989), ("Columbus first reached the Americas", 1492),
         ("the French Revolution began", 1789), ("the United States declared independence", 1776), ("humans first landed on the Moon", 1969),
         ("the Titanic sank", 1912), ("the Soviet Union was dissolved", 1991), ("the Great Fire of London took place", 1666),
         ("the Battle of Hastings was fought", 1066), ("Magna Carta was sealed", 1215), ("the first modern Olympic Games were held", 1896),
         ("the Wright brothers made the first powered flight", 1903), ("the Chernobyl disaster occurred", 1986), ("the Eiffel Tower was completed", 1889),
         ("Napoleon was defeated at Waterloo", 1815), ("the American Civil War ended", 1865), ("the Russian Revolution took place", 1917),
         ("India gained independence", 1947), ("the Black Death reached Europe", 1347), ("Gutenberg printed his Bible", 1455),
         ("the Treaty of Versailles was signed", 1919), ("the Cuban Missile Crisis occurred", 1962), ("Nelson Mandela was released from prison", 1990),
         ("the Euro banknotes and coins were introduced", 2002), ("the Hubble Space Telescope was launched", 1990), ("Mount Vesuvius destroyed Pompeii", 79),
         ("the Wall Street Crash happened", 1929), ("the Suez Canal opened", 1869)]

ARITH = [(12, 12, "*"), (11, 11, "*"), (13, 3, "*"), (15, 4, "*"), (25, 4, "*"), (9, 12, "*"), (7, 15, "*"), (8, 14, "*"), (6, 17, "*"), (5, 23, "*"),
         (14, 14, "*"), (16, 4, "*"), (18, 5, "*"), (21, 3, "*"), (20, 20, "*"), (30, 7, "*"), (40, 6, "*"), (50, 5, "*"), (35, 3, "*"), (45, 2, "*"),
         (64, 36, "+"), (57, 68, "+"), (99, 1, "+"), (123, 456, "+"), (250, 250, "+"), (48, 77, "+"), (81, 19, "+"), (150, 75, "+"), (33, 88, "+"), (200, 5, "+"),
         (144, 12, "/"), (225, 15, "/"), (1000, 8, "/"), (500, 4, "/"), (360, 3, "/"), (999, 9, "/"), (600, 5, "/"), (256, 2, "/"), (900, 9, "/"), (720, 6, "/")]


# ---------------------------------------------------------------- helpers ---------------------------------------
def lead(prompt):
    """the leading space a continuation of `prompt` carries: none after whitespace or an opening quote / bracket"""
    return "" if prompt[-1:] in (" ", "\n", '"', "'", "(", "[", "“", "‘") else " "


def _ok_first(txt):
    return any(ch.isalnum() for ch in txt)


MODE = "multi"  # multi (>= 2 pieces, the H2a set) | single (exactly 1 token) | phrase (>= 2 pieces, every piece a whole word token)


def _is_word_piece(t, lead_space=True):
    """a whole word token: " word" (letters / digits only after the space); the first piece of a string after an
    opening quote has no leading space."""
    if lead_space: return len(t) > 1 and t[0] == " " and t[1:].isalnum()
    return len(t) > 0 and t.isalnum()


def encode_string(tok, string, mode=None):
    """pieces of `string` (>= 2, first piece alphanumeric, round-tripping) or None; retries once without the
    leading space when the first piece is a bare space / punctuation. mode single: exactly one token; mode phrase:
    >= 2 pieces, every piece a whole space-prefixed word token (the first one following the string's own leading
    space)."""
    mode = mode or MODE; s = string
    for _ in range(2):
        ids = tok.encode(s, add_special_tokens=False)
        if mode == "single":
            if len(ids) == 1:
                txt = [tok.decode([t]) for t in ids]
                return (s, ids, txt) if _ok_first(txt[0]) and tok.decode(ids) == s else None
            if len(ids) < 1 or s == s.lstrip() or not tok.decode([ids[0]]).strip() == "": return None
            s = s.lstrip(); continue  # " 7" -> [" ", "7"]: retry as "7"
        if len(ids) < 2: return None
        txt = [tok.decode([t]) for t in ids]
        if mode == "phrase":
            ok = _is_word_piece(txt[0], lead_space=s.startswith(" ")) and all(_is_word_piece(t) for t in txt[1:])
            return (s, ids, txt) if ok and tok.decode(ids) == s else None
        if _ok_first(txt[0]): return (s, ids, txt) if tok.decode(ids) == s else None
        if s == s.lstrip(): return None
        s = s.lstrip()  # " 12" -> [" ", "1", "2"]: retry as "12"
    return None


def item(tok, source, category, kind, prompt, string, target=None, name=""):
    prompt = prompt.rstrip("\n") if prompt.endswith("\n\n") else prompt
    e = encode_string(tok, string)
    if e is None: return None
    s, ids, txt = e
    if s.strip().lower() in prompt.lower(): return None  # would be copying, not recall
    if kind == "bridge" and target is None: return None
    h = hashlib.sha1((prompt + "\x1f" + s).encode()).hexdigest()[:8]
    return {"id": f"{source}:{category}:{h}", "source": source, "category": category, "kind": kind,
            "name": name, "prompt": prompt, "readout": "last_prompt_token", "string": s, "pieces": ids, "pieces_text": txt,
            "target": s.strip() if kind == "answer" else target, "p1_chars": len(txt[0].strip()), "shared_s1_group": txt[0].strip()}


def add(items, it):
    if it is not None: items.append(it)


# ---------------------------------------------------------------- Anthropic sources -----------------------------
def anthropic_items(tok, root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "evaluations", "lens-eval-*.json"))):
        slug = os.path.basename(f)[len("lens-eval-"):-5]
        for it in json.load(open(f))["items"]:
            if isinstance(it["prompt"], list): continue
            pr = it["prompt"]; tgt = it.get("target")
            for s in it["intermediates"]:
                add(out, item(tok, "lens_eval", slug, "bridge", pr, " " + s, target=tgt if tgt else "", name=it.get("name", "")))
            if tgt: add(out, item(tok, "lens_eval", slug + "-target", "answer", pr, lead(pr) + tgt, name=it.get("name", "")))
    for it in json.load(open(os.path.join(root, "experiments", "probe-swap.json")))["items"]:
        pr = it["prompt"]
        add(out, item(tok, "probe_swap", it["category"], "bridge", pr, " " + it["intermediate"], target=it["answer"], name=it["name"]))
        add(out, item(tok, "probe_swap", it["category"] + "-answer", "answer", pr, lead(pr) + it["answer"], name=it["name"]))
    for c in json.load(open(os.path.join(root, "experiments", "flexible-generalization.json")))["categories"]:
        for fn in c["funcs"]:
            for arg in c["args"]:
                pr = fn["template"].format(arg=arg)
                add(out, item(tok, "flexgen", f"{c['name']}-{fn['name']}", "answer", pr, lead(pr) + fn["answers"][arg], name=f"{c['name']}-{fn['name']}-{arg}"))
    return out


# ---------------------------------------------------------------- curated sources -------------------------------
def curated_items(tok):
    out = []; S = "curated"
    cur_count = {}
    for c in COUNTRIES: cur_count[c[2]] = cur_count.get(c[2], 0) + 1
    for country, cap, cur, lang, cont in COUNTRIES:
        if cap: add(out, item(tok, S, "capital", "answer", f"Fact: The capital of {country} is", " " + cap))
        if cap and country.lower() not in cap.lower(): add(out, item(tok, S, "country_of_capital", "answer", f"Fact: {cap} is the capital of", " " + country))
        if cur: add(out, item(tok, S, "currency", "answer", f"Fact: The unit of currency in {country} is the", " " + cur))
        if isinstance(lang, str): add(out, item(tok, S, "language", "answer", f"Fact: The official language of {country} is", " " + lang))
        # bridges: the country is hidden behind its capital or its (unique in the table) currency
        if cap and country.lower() not in cap.lower():
            if cur: add(out, item(tok, S, "bridge_capital_currency", "bridge", f"Fact: The currency of the country whose capital is {cap} is the", " " + country, target=cur))
            if cont: add(out, item(tok, S, "bridge_capital_continent", "bridge", f"Fact: The continent of the country whose capital is {cap} is", " " + country, target=cont))
            if lang: add(out, item(tok, S, "bridge_capital_language", "bridge", f"Fact: The official language of the country whose capital is {cap} is", " " + country, target=lang))
        if cur and cur_count[cur] == 1 and country.lower() not in cur.lower():
            if cap: add(out, item(tok, S, "bridge_currency_capital", "bridge", f"Fact: The capital of the country whose currency is the {cur} is", " " + country, target=cap))
            if cont: add(out, item(tok, S, "bridge_currency_continent", "bridge", f"Fact: The continent of the country whose currency is the {cur} is", " " + country, target=cont))
    for desc, surname, nat, lang in PEOPLE:
        add(out, item(tok, S, "surname", "answer", f"Fact: The surname of {desc} is", " " + surname))
        add(out, item(tok, S, "bridge_person_nationality", "bridge", f"Fact: The nationality of {desc} was", " " + surname, target=nat))
        if lang: add(out, item(tok, S, "bridge_person_language", "bridge", f"Fact: The native language of {desc} was", " " + surname, target=lang))
    for sym, name, z in ELEMENTS:
        add(out, item(tok, S, "element", "answer", f"Fact: The chemical element with the symbol {sym} is", " " + name))
        add(out, item(tok, S, "bridge_element_number", "bridge", f"Fact: The atomic number of the element with the symbol {sym} is", " " + name, target=str(z)))
    for pr, animal in ANIMALS: add(out, item(tok, S, "animal", "answer", "Fact: " + pr, " " + animal))
    for row in TRANSLATIONS:
        for lname, j in LANGS:
            if row[j]: add(out, item(tok, S, f"translation_{lname.lower()}", "answer", f'Fact: The {lname} word for "{row[0]}" is "', row[j]))
    for a, b, op in NUMWORD_PROBLEMS:
        r = a + b if op == "+" else a * b
        add(out, item(tok, S, "number_word", "answer", f"Fact: {numword(a).capitalize()} {'plus' if op == '+' else 'times'} {numword(b)} equals", " " + numword(r)))
    for name, f in FORMULAS: add(out, item(tok, S, "formula", "answer", f"Fact: The chemical formula of {name} is", " " + f))
    for ev, y in YEARS: add(out, item(tok, S, "year", "answer", f"Fact: {ev[0].upper() + ev[1:]} in the year ", str(y)))
    for a, b, op in ARITH:
        r = {"*": a * b, "+": a + b, "/": a // b}[op]
        add(out, item(tok, S, "arith_digits", "answer", f"{a} {op} {b} = ", str(r)))
    return out


# ---------------------------------------------------------------- word-phrase tables (--phrases) ---------------
# Multi-word countries / territories: (name, capital, currency, continent, main language). None = ambiguous.
PHRASE_COUNTRIES = [
    ("South Africa", "Pretoria", "rand", "Africa", None), ("South Korea", "Seoul", "won", "Asia", "Korean"),
    ("North Korea", "Pyongyang", "won", "Asia", "Korean"), ("South Sudan", "Juba", "pound", "Africa", "English"),
    ("New Zealand", "Wellington", "dollar", "Oceania", "English"), ("Sri Lanka", "Colombo", "rupee", "Asia", None),
    ("Saudi Arabia", "Riyadh", "riyal", "Asia", "Arabic"), ("Costa Rica", "San José", "colón", "North America", "Spanish"),
    ("El Salvador", "San Salvador", "dollar", "North America", "Spanish"), ("Czech Republic", "Prague", "koruna", "Europe", "Czech"),
    ("North Macedonia", "Skopje", "denar", "Europe", "Macedonian"), ("Burkina Faso", "Ouagadougou", "franc", "Africa", "French"),
    ("Sierra Leone", "Freetown", "leone", "Africa", "English"), ("Dominican Republic", "Santo Domingo", "peso", "North America", "Spanish"),
    ("Puerto Rico", "San Juan", "dollar", "North America", "Spanish"), ("United States", "Washington", "dollar", "North America", "English"),
    ("United Kingdom", "London", "pound", "Europe", "English"), ("United Arab Emirates", "Abu Dhabi", "dirham", "Asia", "Arabic"),
    ("Papua New Guinea", "Port Moresby", "kina", "Oceania", None), ("Cape Verde", "Praia", "escudo", "Africa", "Portuguese"),
    ("East Timor", "Dili", "dollar", "Asia", None), ("Ivory Coast", "Yamoussoukro", "franc", "Africa", "French"),
    ("Equatorial Guinea", "Malabo", "franc", "Africa", "Spanish"), ("Central African Republic", "Bangui", "franc", "Africa", "French"),
    ("Solomon Islands", "Honiara", "dollar", "Oceania", "English"), ("Marshall Islands", "Majuro", "dollar", "Oceania", None),
    ("Saint Lucia", "Castries", "dollar", "North America", "English"), ("Trinidad and Tobago", "Port of Spain", "dollar", "North America", "English"),
    ("Bosnia and Herzegovina", "Sarajevo", None, "Europe", None), ("San Marino", None, "euro", "Europe", "Italian"),
    ("Vatican City", None, "euro", "Europe", "Italian"), ("Hong Kong", None, "dollar", "Asia", None),
    ("Western Sahara", None, None, "Africa", None), ("Antigua and Barbuda", None, "dollar", "North America", "English"),
    ("Saint Kitts and Nevis", None, "dollar", "North America", "English"), ("Sao Tome and Principe", None, "dobra", "Africa", "Portuguese"),
]
# US states with a multi-word name: (state, capital, nickname, largest city); every state with a multi-word capital.
PHRASE_STATES = [
    ("New York", "Albany", "Empire State", None), ("New Jersey", "Trenton", "Garden State", "Newark"),
    ("New Mexico", "Santa Fe", "Land of Enchantment", "Albuquerque"), ("New Hampshire", "Concord", "Granite State", "Manchester"),
    ("North Carolina", "Raleigh", "Tar Heel State", "Charlotte"), ("South Carolina", "Columbia", "Palmetto State", "Charleston"),
    ("North Dakota", "Bismarck", "Peace Garden State", "Fargo"), ("South Dakota", "Pierre", "Mount Rushmore State", "Sioux Falls"),
    ("West Virginia", "Charleston", "Mountain State", None), ("Rhode Island", "Providence", "Ocean State", None),
    ("Utah", "Salt Lake City", "Beehive State", None), ("Oklahoma", "Oklahoma City", "Sooner State", None),
    ("Missouri", "Jefferson City", "Show-Me State", "Kansas City"), ("Nevada", "Carson City", "Silver State", "Las Vegas"),
    ("Louisiana", "Baton Rouge", "Pelican State", "New Orleans"), ("Arkansas", "Little Rock", "Natural State", None),
    ("Iowa", "Des Moines", "Hawkeye State", None), ("California", "Sacramento", "Golden State", "Los Angeles"),
    ("Texas", "Austin", "Lone Star State", "Houston"), ("Florida", "Tallahassee", "Sunshine State", "Jacksonville"),
]
# (prompt, phrase) answer items; the phrase must be the model's greedy continuation (gated at run time).
PHRASE_PLACES = [
    ("Fact: The most populous city in the United States is", "New York"), ("Fact: The capital of India is", "New Delhi"),
    ("Fact: The largest city in Louisiana is", "New Orleans"), ("Fact: The largest city in California is", "Los Angeles"),
    ("Fact: The largest city in Nevada is", "Las Vegas"), ("Fact: The city where the Golden Gate Bridge stands is", "San Francisco"),
    ("Fact: The Texas city famous for the Alamo is", "San Antonio"), ("Fact: The southernmost large city of California, on the Mexican border, is", "San Diego"),
    ("Fact: The capital of Argentina is", "Buenos Aires"), ("Fact: The capital of Malaysia is", "Kuala Lumpur"),
    ("Fact: The capital of Ethiopia is", "Addis Ababa"), ("Fact: The capital of Cambodia is", "Phnom Penh"),
    ("Fact: The capital of the United Arab Emirates is", "Abu Dhabi"), ("Fact: The largest city of the United Arab Emirates is", "Dubai"),
    ("Fact: The largest city in Israel is", "Tel Aviv"), ("Fact: The legislative capital of South Africa is", "Cape Town"),
    ("Fact: The Brazilian city famous for its Carnival and the Christ the Redeemer statue is", "Rio de Janeiro"),
    ("Fact: The largest city in Brazil is", "São Paulo"), ("Fact: The capital of Mexico is", "Mexico City"),
    ("Fact: The capital of Guatemala is", "Guatemala City"), ("Fact: The capital of Panama is", "Panama City"),
    ("Fact: The capital of the Dominican Republic is", "Santo Domingo"), ("Fact: The capital of Papua New Guinea is", "Port Moresby"),
    ("Fact: The capital of Vietnam is", "Hanoi"), ("Fact: The largest city in Vietnam is", "Ho Chi Minh City"),
    ("Fact: The largest city in Missouri is", "Kansas City"), ("Fact: The capital of Utah is", "Salt Lake City"),
    ("Fact: The former British colony returned to China in 1997 is", "Hong Kong"), ("Fact: The largest city in Scotland is", "Glasgow"),
    ("Fact: The city in Nevada famous for its casinos is", "Las Vegas"), ("Fact: The capital of Sri Lanka is", "Colombo"),
    ("Fact: The largest city of Pakistan is", "Karachi"), ("Fact: The capital of Saudi Arabia is", "Riyadh"),
    ("Fact: The largest city in Canada is", "Toronto"), ("Fact: The largest city in Australia is", "Sydney"),
    ("Fact: The largest city in Turkey is", "Istanbul"), ("Fact: The capital of Western Australia is", "Perth"),
    ("Fact: The largest city in New Zealand is", "Auckland"), ("Fact: The largest city in Switzerland is", "Zurich"),
    ("Fact: The largest city in Morocco is", "Casablanca"), ("Fact: The Indian city formerly called Bombay is", "Mumbai"),
    ("Fact: The Spanish city with the Sagrada Família is", "Barcelona"), ("Fact: The Italian city famous for its canals is", "Venice"),
    ("Fact: The Russian city formerly called Leningrad is", "Saint Petersburg"), ("Fact: The German city famous for Oktoberfest is", "Munich"),
    ("Fact: The Japanese city that was the imperial capital before Tokyo is", "Kyoto"), ("Fact: The Chinese city famous for the Bund is", "Shanghai"),
    ("Fact: The US state whose capital is Santa Fe is", "New Mexico"), ("Fact: The region in northern Spain whose capital is Bilbao is the", "Basque Country"),
    ("Fact: The Caribbean island that is a US territory with capital San Juan is", "Puerto Rico"), ("Fact: The continent of Brazil is", "South America"),
    ("Fact: The continent of Canada is", "North America"), ("Fact: The continent of Egypt is", "Africa"),
    ("Fact: The part of the United Kingdom whose capital is Cardiff is", "Wales"), ("Fact: The part of the United Kingdom whose capital is Belfast is", "Northern Ireland"),
    ("Fact: The Palestinian territory west of the Jordan River is the", "West Bank"), ("Fact: The Indian state whose capital is Kolkata is", "West Bengal"),
    ("Fact: The Australian state whose capital is Sydney is", "New South Wales"), ("Fact: The Australian state whose capital is Perth is", "Western Australia"),
    ("Fact: The Australian state whose capital is Adelaide is", "South Australia"), ("Fact: The Canadian province whose capital is Victoria is", "British Columbia"),
    ("Fact: The Canadian province whose capital is Halifax is", "Nova Scotia"), ("Fact: The Canadian province whose capital is Fredericton is", "New Brunswick"),
    ("Fact: The international organisation headquartered in New York with 193 member states is the", "United Nations"),
    ("Fact: The country whose capital is Washington is the", "United States"), ("Fact: The country whose capital is London is the", "United Kingdom"),
]
PHRASE_GEO = [
    ("Fact: The highest mountain on Earth is", "Mount Everest"), ("Fact: The highest mountain in Africa is", "Mount Kilimanjaro"),
    ("Fact: The highest mountain in Japan is", "Mount Fuji"), ("Fact: The highest mountain in the Alps is", "Mont Blanc"),
    ("Fact: The largest ocean on Earth is the", "Pacific Ocean"), ("Fact: The ocean between Europe and the Americas is the", "Atlantic Ocean"),
    ("Fact: The ocean east of Africa and south of India is the", "Indian Ocean"), ("Fact: The ocean around the North Pole is the", "Arctic Ocean"),
    ("Fact: The sea between Europe and North Africa is the", "Mediterranean Sea"), ("Fact: The sea between Egypt and Saudi Arabia is the", "Red Sea"),
    ("Fact: The sea between Japan and Korea is the", "Sea of Japan"), ("Fact: The sea between Britain and Scandinavia is the", "North Sea"),
    ("Fact: The sea between Turkey and Ukraine is the", "Black Sea"), ("Fact: The largest lake in Africa is", "Lake Victoria"),
    ("Fact: The deepest lake in the world is", "Lake Baikal"), ("Fact: The largest of the North American Great Lakes is", "Lake Superior"),
    ("Fact: The waterfall on the border of Zambia and Zimbabwe is", "Victoria Falls"), ("Fact: The waterfall on the border of the United States and Canada is", "Niagara Falls"),
    ("Fact: The strait between Spain and Morocco is the", "Strait of Gibraltar"), ("Fact: The gulf between Iran and the Arabian Peninsula is the", "Persian Gulf"),
    ("Fact: The bay east of India is the", "Bay of Bengal"), ("Fact: The gulf south of the United States and east of Mexico is the", "Gulf of Mexico"),
    ("Fact: The largest coral reef system in the world is the", "Great Barrier Reef"), ("Fact: The canyon in Arizona carved by the Colorado River is the", "Grand Canyon"),
    ("Fact: The largest hot desert in the world is the", "Sahara"), ("Fact: The rainforest in the Amazon basin is the", "Amazon"),
    ("Fact: The canal connecting the Mediterranean and the Red Sea is the", "Suez Canal"), ("Fact: The canal connecting the Atlantic and the Pacific through Central America is the", "Panama Canal"),
    ("Fact: The wall built across northern China against invaders is the", "Great Wall"), ("Fact: The iron tower in Paris built for the 1889 exposition is the", "Eiffel Tower"),
    ("Fact: The clock tower at the Palace of Westminster in London is", "Big Ben"), ("Fact: The white marble mausoleum in Agra built by Shah Jahan is the", "Taj Mahal"),
    ("Fact: The statue with a lion's body and a human head near the pyramids of Giza is the", "Great Sphinx"), ("Fact: The statue in New York harbour given by France is the", "Statue of Liberty"),
    ("Fact: The suspension bridge across the entrance to San Francisco Bay is the", "Golden Gate Bridge"), ("Fact: The large square in Beijing next to the Forbidden City is", "Tiananmen Square"),
    ("Fact: The square in Moscow beside the Kremlin is", "Red Square"), ("Fact: The official residence of the US president is the", "White House"),
    ("Fact: The London residence of the British monarch is", "Buckingham Palace"), ("Fact: The Inca citadel in the Peruvian Andes is", "Machu Picchu"),
    ("Fact: The temple complex in Cambodia that is the largest religious monument in the world is", "Angkor Wat"), ("Fact: The tallest building in the world, in Dubai, is the", "Burj Khalifa"),
    ("Fact: The famous opera house on Sydney harbour is the", "Sydney Opera House"), ("Fact: The tower in Pisa famous for its tilt is the", "Leaning Tower"),
    ("Fact: The prison island in San Francisco Bay is", "Alcatraz"), ("Fact: The ancient stone circle in Wiltshire, England is", "Stonehenge"),
    ("Fact: The park in Manhattan between the Upper East and Upper West sides is", "Central Park"), ("Fact: The first national park in the United States is", "Yellowstone"),
    ("Fact: The rock formation in central Australia sacred to Aboriginal people is", "Uluru"), ("Fact: The longest river in Africa is the", "Nile"),
    ("Fact: The longest river in the United States is the", "Missouri"), ("Fact: The river that flows through London is the", "Thames"),
    ("Fact: The mountain range that separates Europe from Asia is the", "Ural Mountains"), ("Fact: The mountain range along the west coast of South America is the", "Andes"),
    ("Fact: The mountain range in the western United States that includes Denver is the", "Rocky Mountains"),
]
PHRASE_EVENTS = [
    ("Fact: The global war fought from 1939 to 1945 was", "World War II"), ("Fact: The global war fought from 1914 to 1918 was", "World War I"),
    ("Fact: The geopolitical rivalry between the United States and the Soviet Union after 1945 is called the", "Cold War"),
    ("Fact: The pandemic that killed a third of Europe's population in the 14th century is called the", "Black Death"),
    ("Fact: The worldwide economic collapse that began with the 1929 crash is the", "Great Depression"),
    ("Fact: The 18th-century transition from hand production to machines is called the", "Industrial Revolution"),
    ("Fact: The 1789 uprising that overthrew the French monarchy is the", "French Revolution"), ("Fact: The 1917 uprising that overthrew the Russian tsar is the", "Russian Revolution"),
    ("Fact: The war of 1861 to 1865 between the Union and the Confederacy is the", "American Civil War"),
    ("Fact: The war in which the thirteen colonies won independence from Britain is the", "American Revolution"),
    ("Fact: The 1962 confrontation over Soviet missiles in Cuba is the", "Cuban Missile Crisis"),
    ("Fact: The 1666 fire that destroyed much of the City of London is the", "Great Fire of London"),
    ("Fact: The 1215 charter that limited the power of the English king is", "Magna Carta"), ("Fact: The 1919 treaty that formally ended World War I is the", "Treaty of Versailles"),
    ("Fact: The 1066 battle in which William of Normandy conquered England is the", "Battle of Hastings"),
    ("Fact: The 1815 battle in which Napoleon was finally defeated is the", "Battle of Waterloo"),
    ("Fact: The 1945 meeting of Roosevelt, Churchill and Stalin in Crimea was the", "Yalta Conference"),
    ("Fact: The European cultural movement that began in 14th-century Italy is the", "Renaissance"),
    ("Fact: The 18th-century intellectual movement emphasising reason is the", "Enlightenment"),
    ("Fact: The 1519 to 1522 voyage that first circled the globe was led by", "Ferdinand Magellan"),
    ("Fact: The 1969 mission that first landed humans on the Moon was", "Apollo 11"), ("Fact: The 1986 nuclear accident in Ukraine happened at", "Chernobyl"),
    ("Fact: The 1912 ship that sank after hitting an iceberg was the", "Titanic"), ("Fact: The 1773 protest in which tea was dumped into Boston harbour is the", "Boston Tea Party"),
    ("Fact: The 1955 to 1975 war in Southeast Asia involving the United States is the", "Vietnam War"),
    ("Fact: The 1950 to 1953 war on the Korean peninsula is the", "Korean War"), ("Fact: The 1991 war to expel Iraq from Kuwait is the", "Gulf War"),
    ("Fact: The 1898 war between Spain and the United States is the", "Spanish-American War"),
    ("Fact: The medieval Christian military expeditions to the Holy Land are the", "Crusades"),
    ("Fact: The 1848 discovery that drew thousands of prospectors to California is the", "Gold Rush"),
]
PHRASE_CONCEPTS = {  # category -> [(prompt, phrase)]
    'concept_general': [
        ('Fact: A region of spacetime from which nothing, not even light, can escape is a', 'black hole'),
        ('Fact: The frozen dessert made from cream, sugar and flavouring is', 'ice cream'),
        ('Fact: The head of government of the United Kingdom is the', 'prime minister'),
    ],
    'chemistry': [
        ('Fact: The gas that plants absorb from the air for photosynthesis is', 'carbon dioxide'),
        ('Fact: The poisonous gas produced by incomplete combustion is', 'carbon monoxide'),
        ('Fact: The acid used in car batteries is', 'sulfuric acid'),
        ('Fact: The chemical name of table salt is', 'sodium chloride'),
        ('Fact: The acid that gives vinegar its sour taste is', 'acetic acid'),
        ('Fact: The acid produced in the human stomach is', 'hydrochloric acid'),
        ('Fact: The compound with the formula H2O2 is', 'hydrogen peroxide'),
        ('Fact: The compound with the formula NaHCO3, baking soda, is', 'sodium bicarbonate'),
        ('Fact: The compound with the formula CaCO3 is', 'calcium carbonate'),
        ('Fact: The gas with the formula N2O, also called laughing gas, is', 'nitrous oxide'),
        ('Fact: The compound with the formula NaOH is', 'sodium hydroxide'),
        ('Fact: The compound with the formula KCl is', 'potassium chloride'),
        ('Fact: The compound with the formula MgO is', 'magnesium oxide'),
        ('Fact: The compound with the formula SO2 is', 'sulfur dioxide'),
        ('Fact: The compound with the formula HNO3 is', 'nitric acid'),
        ('Fact: The compound with the formula NH3 is', 'ammonia'),
    ],
    'science': [
        ('Fact: The theory that the universe expanded from a hot, dense state is the', 'Big Bang'),
        ('Fact: The galaxy that contains our solar system is the', 'Milky Way'),
        ('Fact: The organ that pumps blood through the body is the', 'heart'),
        ('Fact: The largest organ of the human body is the', 'skin'),
        ('Fact: The part of the eye that controls how much light enters is the', 'iris'),
        ('Fact: The molecule that carries genetic information in cells is', 'DNA'),
        ('Fact: The powerhouse of the cell is the', 'mitochondrion'),
        ('Fact: The process by which plants make food from sunlight is', 'photosynthesis'),
        ('Fact: The unit of electrical resistance is the', 'ohm'),
        ('Fact: The medical condition in which blood sugar is too high is', 'diabetes'),
        ('Fact: The chemical element with the symbol Fe is', 'iron'),
        ('Fact: The hardest natural substance is', 'diamond'),
        ('Fact: The common name of the disease caused by the varicella virus is', 'chickenpox'),
        ('Fact: The most spoken language in Brazil is', 'Portuguese'),
        ('Fact: The currency of Japan is the', 'yen'),
        ('Fact: The most common blood type is', 'O positive'),
        ('Fact: The study of earthquakes is', 'seismology'),
        ("Fact: The branch of mathematics dealing with triangles' sides and angles is", 'trigonometry'),
        ('Fact: The planet known as the Red Planet is', 'Mars'),
        ('Fact: The largest planet in the solar system is', 'Jupiter'),
        ('Fact: The planet with prominent rings is', 'Saturn'),
        ('Fact: The closest star to Earth after the Sun is', 'Proxima Centauri'),
        ('Fact: The telescope launched into orbit in 1990 is the', 'Hubble Space Telescope'),
        ('Fact: The US space agency is', 'NASA'),
        ('Fact: The first artificial satellite, launched in 1957, was', 'Sputnik'),
        ('Fact: The particle accelerator near Geneva is the', 'Large Hadron Collider'),
        ("Fact: The most abundant gas in Earth's atmosphere is", 'nitrogen'),
        ('Fact: The layer of gas that protects Earth from ultraviolet radiation is the', 'ozone layer'),
        ('Fact: The warming of the planet caused by greenhouse gases is', 'global warming'),
        ('Fact: The energy stored in the nucleus of an atom is', 'nuclear energy'),
        ('Fact: The machine that converts wind into electricity is a', 'wind turbine'),
        ('Fact: The vehicle that runs on electricity instead of petrol is an', 'electric car'),
    ],
    'animal': [
        ('Fact: The largest animal that has ever lived is the', 'blue whale'),
        ('Fact: The largest land carnivore, living in the Arctic, is the', 'polar bear'),
        ('Fact: The bird that is the national symbol of the United States is the', 'bald eagle'),
        ('Fact: The fastest bird, reaching over 300 km/h in a dive, is the', 'peregrine falcon'),
        ('Fact: The bamboo-eating bear native to China is the', 'giant panda'),
        ('Fact: The largest species of penguin is the', 'emperor penguin'),
        ('Fact: The largest fish in the sea, a filter feeder, is the', 'whale shark'),
        ('Fact: The largest living lizard, found in Indonesia, is the', 'Komodo dragon'),
        ('Fact: The large shark made famous by the film Jaws is the', 'great white shark'),
        ('Fact: The venomous spider with a red hourglass marking is the', 'black widow'),
        ('Fact: The dog breed from Germany widely used by police is the', 'German Shepherd'),
        ('Fact: The dog breed from Newfoundland popular as a family pet is the', 'Labrador Retriever'),
        ('Fact: The very large dog breed sometimes called the Apollo of dogs is the', 'Great Dane'),
        ('Fact: The tallest tree species in the world is the', 'coast redwood'),
    ],
    'sport': [
        ('Fact: The sport played with a quarterback and touchdowns is', 'American football'),
        ('Fact: The sport played at Wimbledon is', 'tennis'),
        ('Fact: The board game with kings, queens and knights is', 'chess'),
        ('Fact: The card game where players aim for 21 is', 'blackjack'),
        ('Fact: The Olympic sport of swimming, cycling and running in one race is the', 'triathlon'),
        ('Fact: The mixed martial art from Brazil based on ground fighting is', 'Brazilian Jiu-Jitsu'),
        ('Fact: The team sport played on ice with a puck is', 'ice hockey'),
        ('Fact: The sport of riding waves on a board is', 'surfing'),
        ('Fact: The dance from Argentina performed by a couple in close embrace is the', 'tango'),
        ('Fact: The Cuban dance and music style popular in the 1990s is', 'salsa'),
        ('Fact: The classical dance form with pointe shoes and tutus is', 'ballet'),
        ('Fact: The Brazilian martial art that mixes dance and acrobatics is', 'capoeira'),
        ('Fact: The Japanese martial art using throws and holds is', 'judo'),
        ('Fact: The Korean martial art known for its kicks is', 'taekwondo'),
    ],
    'arts': [
        ("Fact: Leonardo da Vinci's portrait of a woman with an enigmatic smile is the", 'Mona Lisa'),
        ("Fact: Van Gogh's painting of a swirling night sky is", 'The Starry Night'),
        ("Fact: Shakespeare's tragedy about two young lovers from feuding families is", 'Romeo and Juliet'),
        ("Fact: Tolkien's trilogy about the One Ring is", 'The Lord of the Rings'),
        ("Fact: Orwell's satirical novel about farm animals who overthrow their farmer is", 'Animal Farm'),
        ("Fact: Jane Austen's novel about Elizabeth Bennet and Mr Darcy is", 'Pride and Prejudice'),
        ("Fact: Tolstoy's novel about Russia during Napoleon's invasion is", 'War and Peace'),
        ("Fact: Dostoevsky's novel about the student Raskolnikov is", 'Crime and Punishment'),
        ("Fact: Melville's novel about Captain Ahab and a white whale is", 'Moby Dick'),
        ("Fact: Cervantes's novel about a knight who fights windmills is", 'Don Quixote'),
        ("Fact: Vivaldi's set of four violin concertos is", 'The Four Seasons'),
        ("Fact: Tchaikovsky's ballet about a princess turned into a swan is", 'Swan Lake'),
        ("Fact: Tchaikovsky's ballet performed at Christmas is", 'The Nutcracker'),
        ("Fact: The Beatles' 1969 album with the band on a zebra crossing is", 'Abbey Road'),
        ("Fact: Beethoven's choral finale of the Ninth Symphony is the", 'Ode to Joy'),
        ("Fact: Homer's epic about the wanderings of Odysseus is the", 'Odyssey'),
        ('Fact: The 1977 George Lucas space film with Luke Skywalker is', 'Star Wars'),
        ('Fact: The book and film series about a boy wizard is', 'Harry Potter'),
        ('Fact: The 1993 Spielberg film about cloned dinosaurs is', 'Jurassic Park'),
        ('Fact: The 1994 film about a slow-witted man who meets several presidents is', 'Forrest Gump'),
        ('Fact: The 1939 film in which Dorothy travels to Oz is', 'The Wizard of Oz'),
        ('Fact: The 1972 Coppola film about the Corleone family is', 'The Godfather'),
        ('Fact: The 1994 Disney film about the lion cub Simba is', 'The Lion King'),
        ('Fact: The Marvel superhero with a star-spangled shield is', 'Captain America'),
        ('Fact: The Marvel superhero in a powered suit of armour is', 'Iron Man'),
        ('Fact: The detective who lives at 221B Baker Street is', 'Sherlock Holmes'),
        ('Fact: The hobbit who carries the ring to Mount Doom is', 'Frodo Baggins'),
        ('Fact: The captain of the Pequod in Moby-Dick is', 'Captain Ahab'),
        ('Fact: The Disney cartoon mouse created in 1928 is', 'Mickey Mouse'),
        ('Fact: The Disney cartoon duck in a sailor suit is', 'Donald Duck'),
        ("Fact: The cartoon rabbit who says 'What's up, Doc?' is", 'Bugs Bunny'),
        ('Fact: The animated family from Springfield with yellow skin is', 'The Simpsons'),
        ('Fact: The fairy in Peter Pan is', 'Tinker Bell'),
        ('Fact: The lion in The Chronicles of Narnia is', 'Aslan'),
    ],
    'holiday': [
        ('Fact: The US holiday on 4 July is', 'Independence Day'),
        ('Fact: The US holiday on the first Monday of September honouring workers is', 'Labor Day'),
        ('Fact: The US holiday on 11 November honouring military veterans is', 'Veterans Day'),
        ('Fact: The Chinese festival that marks the lunar new year is', 'Chinese New Year'),
        ('Fact: The US holiday on the fourth Thursday of November is', 'Thanksgiving'),
        ('Fact: The Christian festival on 25 December is', 'Christmas'),
        ('Fact: The Hindu festival of lights is', 'Diwali'),
        ('Fact: The holiday on 31 October with costumes and pumpkins is', 'Halloween'),
        ('Fact: The seven-day Jewish festival commemorating the exodus from Egypt is', 'Passover'),
    ],
    'technology': [
        ('Fact: The device that uses satellites to determine location is', 'GPS'),
        ('Fact: The network of computers connected worldwide is the', 'Internet'),
        ('Fact: The technique of computers learning patterns from data is', 'machine learning'),
        ('Fact: The branch of AI that builds neural networks with many layers is', 'deep learning'),
        ('Fact: The most popular programming language for data science is', 'Python'),
        ('Fact: The company that makes the Windows operating system is', 'Microsoft'),
        ('Fact: The search engine company founded by Larry Page and Sergey Brin is', 'Google'),
        ('Fact: The online retailer founded by Jeff Bezos is', 'Amazon'),
        ('Fact: The social network founded by Mark Zuckerberg is', 'Facebook'),
        ('Fact: The electric car company led by Elon Musk is', 'Tesla'),
        ('Fact: The video-sharing website owned by Google is', 'YouTube'),
    ],
    'food': [
        ('Fact: The card game played with a 52-card deck for the highest hand is', 'poker'),
        ('Fact: The Japanese art of paper folding is', 'origami'),
        ('Fact: The Japanese dish of vinegared rice with raw fish is', 'sushi'),
        ('Fact: The Italian flat bread with tomato and cheese is', 'pizza'),
        ('Fact: The French pastry shaped like a crescent is the', 'croissant'),
        ('Fact: The Spanish rice dish from Valencia is', 'paella'),
        ('Fact: The Indian flatbread cooked in a tandoor is', 'naan'),
        ('Fact: The Mexican dish of a folded tortilla with filling is a', 'taco'),
        ('Fact: The dish of raw beef or fish sliced thin and served raw in Italy is', 'carpaccio'),
        ('Fact: The German sausage often eaten with mustard is the', 'bratwurst'),
        ('Fact: The hot beverage brewed from roasted beans is', 'coffee'),
        ('Fact: The fermented drink made from grapes is', 'wine'),
        ('Fact: The bubbly wine from a region of France is', 'champagne'),
        ('Fact: The Mexican spirit made from blue agave is', 'tequila'),
        ('Fact: The Scottish spirit distilled from malted barley is', 'whisky'),
    ],
    'nature': [
        ('Fact: The frozen water that falls from clouds as flakes is', 'snow'),
        ('Fact: The colourful arc in the sky after rain is a', 'rainbow'),
        ('Fact: The rotating column of air touching the ground is a', 'tornado'),
        ('Fact: The tropical storm with winds over 119 km/h in the Atlantic is a', 'hurricane'),
        ('Fact: The sudden shaking of the ground is an', 'earthquake'),
        ('Fact: The large wave caused by an undersea earthquake is a', 'tsunami'),
        ('Fact: The mountain that erupts lava is a', 'volcano'),
        ('Fact: The line of latitude that divides Earth into northern and southern halves is the', 'equator'),
        ('Fact: The imaginary line at 0 degrees longitude through Greenwich is the', 'prime meridian'),
        ('Fact: The northernmost point on Earth is the', 'North Pole'),
        ('Fact: The southernmost point on Earth is the', 'South Pole'),
        ('Fact: The period of 100 years is a', 'century'),
    ],
    'misc': [
        ('Fact: The star sign for people born in late July to August is', 'Leo'),
        ('Fact: The seventh month of the year is', 'July'),
        ('Fact: The second day of the working week is', 'Tuesday'),
        ('Fact: The season between summer and winter is', 'autumn'),
        ('Fact: The ninth month of the year is', 'September'),
        ('Fact: The colour of a ripe banana is', 'yellow'),
        ('Fact: The colours of the flag of Japan are red and', 'white'),
        ('Fact: The three primary colours of light are red, green and', 'blue'),
        ('Fact: The number of days in a leap year is', '366'),
        ('Fact: The number of degrees in a right angle is', '90'),
    ],
    'music': [
        ('Fact: The instrument that measures atmospheric pressure is the', 'barometer'),
        ('Fact: The instrument that measures temperature is the', 'thermometer'),
        ('Fact: The keyboard instrument with 88 keys is the', 'piano'),
        ('Fact: The stringed instrument played with a bow, the smallest of its family, is the', 'violin'),
        ('Fact: The brass instrument with a slide is the', 'trombone'),
        ('Fact: The music genre born in New Orleans with improvisation is', 'jazz'),
        ('Fact: The music genre from Jamaica popularised by Bob Marley is', 'reggae'),
        ('Fact: The music genre from the Bronx with rapping is', 'hip hop'),
        ('Fact: The 1960s British band of Lennon and McCartney is', 'The Beatles'),
        ('Fact: The rock band fronted by Freddie Mercury is', 'Queen'),
        ('Fact: The rock band of Mick Jagger and Keith Richards is', 'The Rolling Stones'),
        ('Fact: The Seattle grunge band fronted by Kurt Cobain is', 'Nirvana'),
        ('Fact: The British band whose album Dark Side of the Moon was a hit is', 'Pink Floyd'),
        ('Fact: The band whose lead singer is Bono is', 'U2'),
    ],
    'game': [
        ('Fact: The card game in which players try to empty their hand by matching colours and numbers is', 'Uno'),
        ('Fact: The video game in which players build with blocks is', 'Minecraft'),
        ('Fact: The Nintendo plumber mascot is', 'Mario'),
        ('Fact: The yellow electric mouse Pokémon is', 'Pikachu'),
        ('Fact: The board game of buying and renting properties is', 'Monopoly'),
        ('Fact: The puzzle cube with coloured faces invented in Hungary is the', "Rubik's Cube"),
        ('Fact: The word game played with lettered tiles on a board is', 'Scrabble'),
        ('Fact: The strategy game from ancient China played with black and white stones is', 'Go'),
    ],
}
# (description of the person, full name, nationality); prompts "Fact: <desc> was" (answer) and
# "Fact: The nationality of <desc> was" (bridge hiding the full name).
PHRASE_PEOPLE = [  # (description, full name, nationality / list of accepted nationalities, category)
    ('The physicist who formulated the laws of motion and universal gravitation', 'Isaac Newton', ['English', 'British'], 'person_science'),
    ('The physicist who developed the theory of relativity', 'Albert Einstein', ['German', 'Swiss', 'American'], 'person_science'),
    ('The scientist who discovered radium and polonium', 'Marie Curie', ['Polish', 'French'], 'person_science'),
    ('The naturalist who proposed evolution by natural selection', 'Charles Darwin', ['English', 'British'], 'person_science'),
    ("The astronomer who first used a telescope to observe Jupiter's moons", 'Galileo Galilei', 'Italian', 'person_science'),
    ('The English playwright who wrote Hamlet and Macbeth', 'William Shakespeare', ['English', 'British'], 'person_writer'),
    ('The English novelist who wrote Oliver Twist and A Christmas Carol', 'Charles Dickens', ['English', 'British'], 'person_writer'),
    ('The English novelist who wrote Pride and Prejudice', 'Jane Austen', ['English', 'British'], 'person_writer'),
    ('The American author who wrote The Adventures of Tom Sawyer', 'Mark Twain', 'American', 'person_writer'),
    ('The first president of the United States', 'George Washington', 'American', 'person_history'),
    ('The US president who issued the Emancipation Proclamation', 'Abraham Lincoln', 'American', 'person_history'),
    ('The main author of the US Declaration of Independence', 'Thomas Jefferson', 'American', 'person_history'),
    ('The American founding father who experimented with lightning and a kite', 'Benjamin Franklin', 'American', 'person_history'),
    ('The British prime minister for most of World War II', 'Winston Churchill', ['British', 'English'], 'person_history'),
    ('The French emperor defeated at Waterloo', 'Napoleon Bonaparte', 'French', 'person_history'),
    ('The Roman general assassinated on the Ides of March in 44 BC', 'Julius Caesar', 'Roman', 'person_history'),
    ('The American civil rights leader who gave the I Have a Dream speech', 'Martin Luther King', 'American', 'person_history'),
    ('The first black president of South Africa', 'Nelson Mandela', 'South African', 'person_history'),
    ("The leader of India's non-violent independence movement", 'Mahatma Gandhi', 'Indian', 'person_history'),
    ('The philosopher who wrote The Communist Manifesto with Engels', 'Karl Marx', 'German', 'person_writer'),
    ('The Austrian founder of psychoanalysis', 'Sigmund Freud', 'Austrian', 'person_science'),
    ('The Serbian-American inventor who pioneered alternating current', 'Nikola Tesla', ['Serbian', 'American'], 'person_science'),
    ('The American inventor of the phonograph and a practical light bulb', 'Thomas Edison', 'American', 'person_science'),
    ('The Scottish scientist who discovered penicillin', 'Alexander Fleming', ['Scottish', 'British'], 'person_science'),
    ('The French chemist who developed pasteurisation and the rabies vaccine', 'Louis Pasteur', 'French', 'person_science'),
    ('The British mathematician who broke the Enigma code at Bletchley Park', 'Alan Turing', ['British', 'English'], 'person_science'),
    ('The English mathematician regarded as the first computer programmer', 'Ada Lovelace', ['English', 'British'], 'person_science'),
    ('The British physicist who wrote A Brief History of Time', 'Stephen Hawking', ['British', 'English'], 'person_science'),
    ('The first person to walk on the Moon', 'Neil Armstrong', 'American', 'person_science'),
    ('The first human to travel into space', 'Yuri Gagarin', ['Soviet', 'Russian'], 'person_science'),
    ('The explorer who reached the Americas in 1492 sailing for Spain', 'Christopher Columbus', ['Italian', 'Genoese'], 'person_history'),
    ('The Venetian merchant who wrote about his travels to China', 'Marco Polo', ['Venetian', 'Italian'], 'person_history'),
    ('The Portuguese explorer who found the sea route to India around Africa', 'Vasco da Gama', 'Portuguese', 'person_history'),
    ('The British navigator who mapped the east coast of Australia in 1770', 'James Cook', ['British', 'English'], 'person_history'),
    ('The American aviator who was the first woman to fly solo across the Atlantic', 'Amelia Earhart', 'American', 'person_history'),
    ('The Spanish painter of Guernica', 'Pablo Picasso', 'Spanish', 'person_arts'),
    ('The French impressionist who painted Water Lilies', 'Claude Monet', 'French', 'person_arts'),
    ('The Spanish surrealist who painted The Persistence of Memory', 'Salvador Dali', 'Spanish', 'person_arts'),
    ('The Mexican painter famous for her self-portraits', 'Frida Kahlo', 'Mexican', 'person_arts'),
    ("The pop artist who painted Campbell's soup cans", 'Andy Warhol', 'American', 'person_arts'),
    ('The founder of the company behind Mickey Mouse', 'Walt Disney', 'American', 'person_arts'),
    ('The co-founder of Apple who introduced the iPhone', 'Steve Jobs', 'American', 'person_science'),
    ('The co-founder of Microsoft', 'Bill Gates', 'American', 'person_science'),
    ('The founder of Amazon', 'Jeff Bezos', 'American', 'person_science'),
    ('The founder of the Ford Motor Company', 'Henry Ford', 'American', 'person_science'),
    ('The Scottish economist who wrote The Wealth of Nations', 'Adam Smith', ['Scottish', 'British'], 'person_writer'),
    ('The English philosopher who wrote Two Treatises of Government', 'John Locke', ['English', 'British'], 'person_writer'),
    ('The Scottish philosopher who wrote A Treatise of Human Nature', 'David Hume', ['Scottish', 'British'], 'person_writer'),
    ('The German philosopher who wrote the Critique of Pure Reason', 'Immanuel Kant', 'German', 'person_writer'),
    ('The German philosopher who wrote Thus Spoke Zarathustra', 'Friedrich Nietzsche', 'German', 'person_writer'),
    ("The French philosopher who said 'I think, therefore I am'", 'Rene Descartes', 'French', 'person_writer'),
    ('The French mathematician after whom a triangle of binomial coefficients is named', 'Blaise Pascal', 'French', 'person_science'),
    ('The English crime writer who created Hercule Poirot', 'Agatha Christie', ['English', 'British'], 'person_writer'),
    ('The English author who wrote Animal Farm and Nineteen Eighty-Four', 'George Orwell', ['English', 'British'], 'person_writer'),
    ('The American author who wrote The Old Man and the Sea', 'Ernest Hemingway', 'American', 'person_writer'),
    ('The English author who wrote Mrs Dalloway and To the Lighthouse', 'Virginia Woolf', ['English', 'British'], 'person_writer'),
    ('The Irish writer who wrote The Picture of Dorian Gray', 'Oscar Wilde', 'Irish', 'person_writer'),
    ('The American poet who wrote The Raven', 'Edgar Allan Poe', 'American', 'person_writer'),
    ('The American poet who wrote Leaves of Grass', 'Walt Whitman', 'American', 'person_writer'),
    ('The Irish author who wrote Ulysses', 'James Joyce', 'Irish', 'person_writer'),
    ('The Colombian author who wrote One Hundred Years of Solitude', 'Gabriel Garcia Marquez', 'Colombian', 'person_writer'),
    ('The Russian author who wrote War and Peace', 'Leo Tolstoy', 'Russian', 'person_writer'),
    ('The Russian author who wrote Crime and Punishment', 'Fyodor Dostoevsky', 'Russian', 'person_writer'),
    ('The German composer of the Ninth Symphony with the Ode to Joy', 'Ludwig van Beethoven', 'German', 'person_arts'),
    ('The German composer of the Brandenburg Concertos', 'Johann Sebastian Bach', 'German', 'person_arts'),
    ('The Austrian composer of The Magic Flute', 'Wolfgang Amadeus Mozart', 'Austrian', 'person_arts'),
    ('The Polish composer of the Nocturnes for piano', 'Frederic Chopin', 'Polish', 'person_arts'),
    ('The Italian composer of The Four Seasons', 'Antonio Vivaldi', 'Italian', 'person_arts'),
    ('The Russian composer of Swan Lake and The Nutcracker', 'Pyotr Tchaikovsky', 'Russian', 'person_arts'),
    ('The American composer of Rhapsody in Blue', 'George Gershwin', 'American', 'person_arts'),
    ('The American basketball player who won six NBA titles with the Chicago Bulls', 'Michael Jordan', 'American', 'person_sport'),
    ('The American boxer born Cassius Clay', 'Muhammad Ali', 'American', 'person_sport'),
    ('The Jamaican sprinter who holds the 100 m world record', 'Usain Bolt', 'Jamaican', 'person_sport'),
    ('The American tennis player who won 23 Grand Slam singles titles', 'Serena Williams', 'American', 'person_sport'),
    ('The Swiss tennis player who won 20 Grand Slam titles', 'Roger Federer', 'Swiss', 'person_sport'),
    ('The Argentine footballer who captained the 2022 World Cup winners', 'Lionel Messi', ['Argentine', 'Argentinian'], 'person_sport'),
    ('The Portuguese footballer who has scored the most international goals', 'Cristiano Ronaldo', 'Portuguese', 'person_sport'),
    ('The American golfer who won the Masters in 1997 at age 21', 'Tiger Woods', 'American', 'person_sport'),
    ('The American swimmer with 23 Olympic gold medals', 'Michael Phelps', 'American', 'person_sport'),
    ('The Brazilian footballer who won three World Cups', 'Pele', 'Brazilian', 'person_sport'),
    ('The singer known as the King of Rock and Roll', 'Elvis Presley', 'American', 'person_arts'),
    ('The singer known as the King of Pop who released Thriller', 'Michael Jackson', 'American', 'person_arts'),
    ('The lead singer of Queen', 'Freddie Mercury', ['British', 'English'], 'person_arts'),
    ('The Beatles member who wrote Imagine', 'John Lennon', ['English', 'British'], 'person_arts'),
    ("The American folk singer who wrote Blowin' in the Wind", 'Bob Dylan', 'American', 'person_arts'),
    ('The English silent-film comedian with a bowler hat and cane', 'Charlie Chaplin', ['English', 'British'], 'person_arts'),
    ('The English film director known as the Master of Suspense', 'Alfred Hitchcock', ['English', 'British'], 'person_arts'),
    ('The American director of Jaws and Jurassic Park', 'Steven Spielberg', 'American', 'person_arts'),
    ('The American actor who played Forrest Gump', 'Tom Hanks', 'American', 'person_arts'),
    ('The American actress who starred in Some Like It Hot', 'Marilyn Monroe', 'American', 'person_arts'),
    ('The Albanian-born nun who worked with the poor in Kolkata', 'Mother Teresa', ['Albanian', 'Indian'], 'person_history'),
    ('The queen of France executed in 1793 during the Revolution', 'Marie Antoinette', ['Austrian', 'French'], 'person_history'),
    ('The British monarch who reigned from 1837 to 1901', 'Queen Victoria', ['British', 'English'], 'person_history'),
    ('The French peasant girl who led armies and was burned at the stake in 1431', 'Joan of Arc', 'French', 'person_history'),
    ('The founder of the Mongol Empire', 'Genghis Khan', ['Mongol', 'Mongolian'], 'person_history'),
    ('The Macedonian king who conquered the Persian Empire', 'Alexander the Great', ['Macedonian', 'Greek'], 'person_history'),
    ('The Greek philosopher who taught Plato', 'Socrates', 'Greek', 'person_writer'),
    ('The physicist who proposed the uncertainty principle', 'Werner Heisenberg', 'German', 'person_science'),
    ('The Danish physicist who proposed the 1913 model of the atom', 'Niels Bohr', 'Danish', 'person_science'),
    ('The Scottish inventor credited with the telephone', 'Alexander Graham Bell', ['Scottish', 'American', 'British'], 'person_science'),
    ('The Italian inventor of the radio telegraph', 'Guglielmo Marconi', 'Italian', 'person_science'),
    ('The English physicist who discovered the electron', 'Joseph John Thomson', ['English', 'British'], 'person_science'),
    ('The New Zealand physicist who discovered the atomic nucleus', 'Ernest Rutherford', ['New Zealand', 'British'], 'person_science'),
    ('The American physicist who led the Manhattan Project', 'Robert Oppenheimer', 'American', 'person_science'),
    ('The English nurse who founded modern nursing during the Crimean War', 'Florence Nightingale', ['English', 'British'], 'person_science'),
    ('The Austrian monk who founded genetics with pea plants', 'Gregor Mendel', ['Austrian', 'Czech'], 'person_science'),
    ('The Swedish chemist who invented dynamite and founded the prizes bearing his name', 'Alfred Nobel', 'Swedish', 'person_science'),
    ('The Swedish botanist who created binomial nomenclature', 'Carl Linnaeus', 'Swedish', 'person_science'),
    ('The German astronomer who discovered the laws of planetary motion', 'Johannes Kepler', 'German', 'person_science'),
    ('The Polish astronomer who placed the Sun at the centre of the solar system', 'Nicolaus Copernicus', 'Polish', 'person_science'),
    ('The Dutch painter of The Starry Night', 'Vincent van Gogh', 'Dutch', 'person_arts'),
    ('The Dutch painter of The Night Watch', 'Rembrandt van Rijn', 'Dutch', 'person_arts'),
    ('The Italian painter of the Sistine Chapel ceiling', 'Michelangelo Buonarroti', 'Italian', 'person_arts'),
    ('The Italian painter of the Mona Lisa', 'Leonardo da Vinci', 'Italian', 'person_arts'),
    ('The Spanish architect of the Sagrada Família', 'Antoni Gaudi', 'Spanish', 'person_arts'),
    ('The French sculptor of The Thinker', 'Auguste Rodin', 'French', 'person_arts'),
    ('The American architect who designed Fallingwater', 'Frank Lloyd Wright', 'American', 'person_arts'),
    ('The English naturalist who presents Planet Earth', 'David Attenborough', ['English', 'British'], 'person_science'),
    ('The Kenyan environmentalist who won the 2004 Nobel Peace Prize', 'Wangari Maathai', 'Kenyan', 'person_science'),
    ("The Pakistani activist for girls' education who won the 2014 Nobel Peace Prize", 'Malala Yousafzai', 'Pakistani', 'person_history'),
    ('The Indian mathematician known for his work on infinite series and the number 1729', 'Srinivasa Ramanujan', 'Indian', 'person_science'),
    ('The Chinese philosopher whose sayings are collected in the Analects', 'Confucius', 'Chinese', 'person_writer'),
    ('The Japanese film director of Seven Samurai', 'Akira Kurosawa', 'Japanese', 'person_arts'),
    ('The Japanese animator who directed Spirited Away', 'Hayao Miyazaki', 'Japanese', 'person_arts'),
    ('The Chilean poet who wrote Twenty Love Poems and a Song of Despair', 'Pablo Neruda', 'Chilean', 'person_writer'),
    ('The Nigerian author of Things Fall Apart', 'Chinua Achebe', 'Nigerian', 'person_writer'),
    ('The Bengali poet who wrote Gitanjali', 'Rabindranath Tagore', ['Indian', 'Bengali'], 'person_writer'),
]
# landmark bridges: (prompt hiding the landmark, landmark string, target country list)
PHRASE_LANDMARK_BRIDGES = [
    ("Fact: The country where the highest mountain on Earth stands, on its border with China, is", "Mount Everest", ["Nepal"]),
    ("Fact: The country where the iron tower built for the 1889 world's fair stands is", "Eiffel Tower", ["France"]),
    ("Fact: The country where the wall built across its northern frontier against nomadic invaders stands is", "Great Wall", ["China"]),
    ("Fact: The country where the Inca citadel high in the Andes is located is", "Machu Picchu", ["Peru"]),
    ("Fact: The country where the largest religious monument in the world, a 12th-century temple complex, stands is", "Angkor Wat", ["Cambodia"]),
    ("Fact: The country where the white marble mausoleum built by Shah Jahan stands is", "Taj Mahal", ["India"]),
    ("Fact: The country where the suspension bridge across the entrance to San Francisco Bay stands is the", "Golden Gate Bridge", ["United States", "USA", "US"]),
    ("Fact: The country where the copper statue given by France stands in a harbour is the", "Statue of Liberty", ["United States", "USA", "US"]),
    ("Fact: The country where the tallest building in the world stands is the", "Burj Khalifa", ["United Arab Emirates", "UAE"]),
    ("Fact: The country where the famous harbourside opera house with sail-shaped roofs stands is", "Sydney Opera House", ["Australia"]),
    ("Fact: The country where the highest mountain in Africa stands is", "Mount Kilimanjaro", ["Tanzania"]),
    ("Fact: The country where the deepest lake in the world lies is", "Lake Baikal", ["Russia"]),
    ("Fact: The country where the canyon carved by the Colorado River lies is the", "Grand Canyon", ["United States", "USA", "US"]),
    ("Fact: The country where the largest coral reef system lies is", "Great Barrier Reef", ["Australia"]),
    ("Fact: The country where the London residence of the monarch stands is the", "Buckingham Palace", ["United Kingdom", "UK", "England"]),
    ("Fact: The country where the official residence of the president is a white mansion on Pennsylvania Avenue is the", "White House", ["United States", "USA", "US"]),
    ("Fact: The country where the square beside the Kremlin lies is", "Red Square", ["Russia"]),
    ("Fact: The country where the square next to the Forbidden City lies is", "Tiananmen Square", ["China"]),
    ("Fact: The country where the sacred red rock formation in the central desert lies is", "Uluru", ["Australia"]),
    ("Fact: The country where the ancient stone circle in Wiltshire stands is", "Stonehenge", ["England", "United Kingdom", "UK"]),
]
# concept bridges: (prompt hiding the compound name, compound string, target formula)
PHRASE_CONCEPT_BRIDGES = [
    ("Fact: The chemical formula of the gas that plants absorb for photosynthesis is", "carbon dioxide", ["CO2", "CO₂"]),
    ("Fact: The chemical formula of the poisonous gas produced by incomplete combustion is", "carbon monoxide", ["CO"]),
    ("Fact: The chemical formula of the acid used in car batteries is", "sulfuric acid", ["H2SO4", "H₂SO₄"]),
    ("Fact: The chemical formula of table salt is", "sodium chloride", ["NaCl"]),
    ("Fact: The chemical formula of the acid that gives vinegar its sour taste is", "acetic acid", ["CH3COOH", "C2H4O2"]),
    ("Fact: The chemical formula of the acid produced in the human stomach is", "hydrochloric acid", ["HCl"]),
    ("Fact: The chemical formula of baking soda is", "sodium bicarbonate", ["NaHCO3", "NaHCO₃"]),
    ("Fact: The chemical formula of the main mineral in limestone and chalk is", "calcium carbonate", ["CaCO3", "CaCO₃"]),
    ("Fact: The chemical formula of laughing gas is", "nitrous oxide", ["N2O", "N₂O"]),
    ("Fact: The year the global war that began in 1939 ended was", "World War II", ["1945"]),
    ("Fact: The year the global war that began in 1914 ended was", "World War I", ["1918"]),
    ("Fact: The century in which the pandemic that killed a third of Europe's population struck was the", "Black Death", ["14th", "fourteenth"]),
    ("Fact: The year the uprising that overthrew the French monarchy began was", "French Revolution", ["1789"]),
    ("Fact: The year the war between the Union and the Confederacy ended was", "American Civil War", ["1865"]),
    ("Fact: The year of the confrontation over Soviet missiles in Cuba was", "Cuban Missile Crisis", ["1962"]),
    ("Fact: The author of the novels about the boy wizard with a lightning-shaped scar is", "Harry Potter", ["Rowling", "J.K. Rowling", "J. K. Rowling"]),
    ("Fact: The director of the 1993 film about cloned dinosaurs on an island was", "Jurassic Park", ["Spielberg", "Steven Spielberg"]),
    ("Fact: The creator of the 1977 space film featuring Luke Skywalker is", "Star Wars", ["George Lucas", "Lucas"]),
    ("Fact: The author of the stories about the detective at 221B Baker Street was", "Sherlock Holmes", ["Arthur Conan Doyle", "Conan Doyle", "Doyle"]),
    ("Fact: The company whose mascot is the cartoon mouse created in 1928 is", "Mickey Mouse", ["Disney", "The Walt Disney Company"]),
    ("Fact: The composer of the ballet about a princess turned into a swan was", "Swan Lake", ["Tchaikovsky", "Pyotr Tchaikovsky"]),
    ("Fact: The author of the novel about Captain Ahab and a white whale was", "Moby Dick", ["Melville", "Herman Melville"]),
    ("Fact: The author of the tragedy about two young lovers from feuding families in Verona was", "Romeo and Juliet", ["Shakespeare", "William Shakespeare"]),
    ("Fact: The author of the trilogy about the One Ring was", "The Lord of the Rings", ["Tolkien", "J.R.R. Tolkien", "J. R. R. Tolkien"]),
    ("Fact: The organ that pumps blood through the body has this many chambers:", "heart", ["four", "4"]),
]

# More word-level phrases (added after the first build dropped most multi-piece surnames): category -> [(prompt, phrase)].
PHRASE_MORE = {
    "region": [
        ("Fact: The region of the United States comprising Maine, Vermont, New Hampshire, Massachusetts, Rhode Island and Connecticut is", "New England"),
        ("Fact: The French overseas territory in the Pacific whose capital is Nouméa is", "New Caledonia"), ("Fact: The large island north of Australia shared by Indonesia and Papua New Guinea is", "New Guinea"),
        ("Fact: The region of California famous for its technology companies is", "Silicon Valley"), ("Fact: The street in Manhattan synonymous with American finance is", "Wall Street"),
        ("Fact: The London street where the prime minister lives at number 10 is", "Downing Street"), ("Fact: The London square with Nelson's Column is", "Trafalgar Square"),
        ("Fact: The New York square famous for its New Year's Eve ball drop is", "Times Square"), ("Fact: The large royal park in central London is", "Hyde Park"),
        ("Fact: The hottest place in North America, a desert valley in California, is", "Death Valley"), ("Fact: The California valley famous for its wine is", "Napa Valley"),
        ("Fact: The region spanning Arabia, the Levant, Iran and Egypt is the", "Middle East"), ("Fact: The island that contains England, Scotland and Wales is", "Great Britain"),
        ("Fact: The five large freshwater lakes on the border of the United States and Canada are the", "Great Lakes"), ("Fact: The ocean that surrounds Antarctica is the", "Southern Ocean"),
        ("Fact: The islands of the Caribbean are also known as the", "West Indies"), ("Fact: The US military academy on the Hudson River is", "West Point"),
        ("Fact: The region of Africa that includes Kenya, Tanzania and Uganda is", "East Africa"), ("Fact: The region of Asia that includes China, Japan and Korea is", "East Asia"),
        ("Fact: The region of Asia that includes Kazakhstan, Uzbekistan and Turkmenistan is", "Central Asia"), ("Fact: The region between Mexico and Colombia that includes Panama and Costa Rica is", "Central America"),
        ("Fact: The region of Europe that includes Poland, Ukraine and Romania is", "Eastern Europe"), ("Fact: The region of Europe that includes France, Germany and the Netherlands is", "Western Europe"),
        ("Fact: The region of Asia that includes Thailand, Vietnam and Indonesia is", "Southeast Asia"), ("Fact: The region of Asia that includes India, Pakistan and Bangladesh is", "South Asia"),
        ("Fact: The northernmost point of the Earth's axis is the", "North Pole"), ("Fact: The southernmost point of the Earth's axis is the", "South Pole"),
        ("Fact: The flat grassland region of the central United States is the", "Great Plains"), ("Fact: The motor race held on the streets of Monaco each year is the Monaco", "Grand Prix"),
        ("Fact: The stadium in Manhattan that hosts the New York Knicks is", "Madison Square Garden"), ("Fact: The elevated railway park in Manhattan built on an old freight line is the", "High Line"),
        ("Fact: The world's largest tropical rainforest, in South America, is the", "Amazon rainforest"), ("Fact: The strait that separates Europe from Asia at Istanbul is the", "Bosphorus"),
        ("Fact: The sea that separates Great Britain from Ireland is the", "Irish Sea"), ("Fact: The channel that separates England from France is the", "English Channel"),
        ("Fact: The bay on the coast of San Francisco is", "San Francisco Bay"), ("Fact: The city in Texas that is the state capital is", "Austin"),
        ("Fact: The US island territory in the Pacific whose capital is Hagåtña is", "Guam"), ("Fact: The most populous US state is", "California"),
        ("Fact: The archipelago in the Atlantic belonging to Portugal is the", "Azores"), ("Fact: The Spanish islands off the coast of Morocco are the", "Canary Islands"),
        ("Fact: The islands off Ecuador famous for Darwin's finches are the", "Galapagos Islands"), ("Fact: The islands in the South Atlantic disputed by Britain and Argentina are the", "Falkland Islands"),
        ("Fact: The British crown dependency in the Irish Sea is the", "Isle of Man"), ("Fact: The Hawaiian island that contains Honolulu is", "Oahu"),
        ("Fact: The US state made up of islands in the Pacific is", "Hawaii"), ("Fact: The largest US state by area is", "Alaska"),
        ("Fact: The country north of the United States is", "Canada"), ("Fact: The strait between Alaska and Russia is the", "Bering Strait"),
        ("Fact: The bay on the east coast of Canada known for the world's highest tides is the", "Bay of Fundy"), ("Fact: The bay in the east of Canada named after explorer Henry Hudson is", "Hudson Bay"),
    ],
    "president": [
        ("Fact: The first president of the United States was", "George Washington"), ("Fact: The second president of the United States was", "John Adams"),
        ("Fact: The US president who made the Louisiana Purchase was", "Thomas Jefferson"), ("Fact: The US president known as the Father of the Constitution was", "James Madison"),
        ("Fact: The US president whose 1823 doctrine opposed European colonisation of the Americas was", "James Monroe"), ("Fact: The seventh US president, hero of the Battle of New Orleans, was", "Andrew Jackson"),
        ("Fact: The US president who led the Union during the Civil War was", "Abraham Lincoln"), ("Fact: The US president who built the Panama Canal and had a teddy bear named after him was", "Theodore Roosevelt"),
        ("Fact: The US president during World War I who proposed the League of Nations was", "Woodrow Wilson"), ("Fact: The US president who launched the New Deal was", "Franklin Roosevelt"),
        ("Fact: The US president who ordered the atomic bombing of Hiroshima was", "Harry Truman"), ("Fact: The US president assassinated in Dallas in 1963 was", "John Kennedy"),
        ("Fact: The US president who signed the Civil Rights Act of 1964 was", "Lyndon Johnson"), ("Fact: The US president who resigned over the Watergate scandal was", "Richard Nixon"),
        ("Fact: The US president who pardoned Richard Nixon was", "Gerald Ford"), ("Fact: The US president who brokered the Camp David Accords was", "Jimmy Carter"),
        ("Fact: The former Hollywood actor who became the 40th US president was", "Ronald Reagan"), ("Fact: The 42nd US president, impeached in 1998, was", "Bill Clinton"),
        ("Fact: The first African American president of the United States was", "Barack Obama"), ("Fact: The 45th president of the United States was", "Donald Trump"),
        ("Fact: The 46th president of the United States was", "Joe Biden"), ("Fact: The British prime minister known as the Iron Lady was", "Margaret Thatcher"),
        ("Fact: The British prime minister who led the Labour Party to victory in 1997 was", "Tony Blair"), ("Fact: The president of Russia since 2012 is", "Vladimir Putin"),
        ("Fact: The chancellor of Germany from 2005 to 2021 was", "Angela Merkel"), ("Fact: The president of France elected in 2017 is", "Emmanuel Macron"),
        ("Fact: The first prime minister of India was", "Jawaharlal Nehru"), ("Fact: The prime minister of India since 2014 is", "Narendra Modi"),
        ("Fact: The founder of the People's Republic of China in 1949 was", "Mao Zedong"), ("Fact: The leader of the Soviet Union during World War II was", "Joseph Stalin"),
        ("Fact: The dictator of Germany from 1933 to 1945 was", "Adolf Hitler"), ("Fact: The Cuban revolutionary who led Cuba from 1959 was", "Fidel Castro"),
        ("Fact: The last leader of the Soviet Union was", "Mikhail Gorbachev"), ("Fact: The first president of Turkey, founder of the republic, was", "Mustafa Kemal Ataturk"),
    ],
    "person_more": [
        ("Fact: The billionaire founder of SpaceX and CEO of Tesla is", "Elon Musk"), ("Fact: The founder of Facebook is", "Mark Zuckerberg"),
        ("Fact: The co-founder of Google with Sergey Brin is", "Larry Page"), ("Fact: The investor known as the Oracle of Omaha is", "Warren Buffett"),
        ("Fact: The founder of Microsoft who later became a philanthropist is", "Bill Gates"), ("Fact: The singer whose albums include 1989 and Folklore is", "Taylor Swift"),
        ("Fact: The singer known for Poker Face and Bad Romance is", "Lady Gaga"), ("Fact: The actor who played Ethan Hunt in Mission: Impossible is", "Tom Cruise"),
        ("Fact: The actor who played Tyler Durden in Fight Club is", "Brad Pitt"), ("Fact: The actor who played the Fresh Prince of Bel-Air is", "Will Smith"),
        ("Fact: The actor who played Jack Sparrow in Pirates of the Caribbean is", "Johnny Depp"), ("Fact: The actor who played Indiana Jones and Han Solo is", "Harrison Ford"),
        ("Fact: The actor who played Tony Stark in the Marvel films is", "Robert Downey Jr"), ("Fact: The actress who played Hermione Granger is", "Emma Watson"),
        ("Fact: The actor who played Harry Potter in the films is", "Daniel Radcliffe"), ("Fact: The actor who played Jack Dawson in Titanic is", "Leonardo DiCaprio"),
        ("Fact: The author of the horror novels The Shining and It is", "Stephen King"), ("Fact: The author of The Da Vinci Code is", "Dan Brown"),
        ("Fact: The author of the Harry Potter books is", "J K Rowling"), ("Fact: The author of A Song of Ice and Fire is", "George Martin"),
        ("Fact: The creator of Sherlock Holmes was", "Arthur Conan Doyle"), ("Fact: The author of The Jungle Book was", "Rudyard Kipling"),
        ("Fact: The English author of Robinson Crusoe was", "Daniel Defoe"), ("Fact: The author of Treasure Island was", "Robert Louis Stevenson"),
        ("Fact: The author of Alice's Adventures in Wonderland was", "Lewis Carroll"), ("Fact: The author of Gulliver's Travels was", "Jonathan Swift"),
        ("Fact: The author of Paradise Lost was", "John Milton"), ("Fact: The poet who wrote Ode to a Nightingale was", "John Keats"),
        ("Fact: The poet who wrote The Road Not Taken was", "Robert Frost"), ("Fact: The American poet who wrote The Waste Land was", "T S Eliot"),
        ("Fact: The English poet who wrote Paradise Lost was", "John Milton"), ("Fact: The author of The Great Gatsby was", "Scott Fitzgerald"),
        ("Fact: The author of To Kill a Mockingbird was", "Harper Lee"), ("Fact: The author of The Catcher in the Rye was", "J D Salinger"),
        ("Fact: The author of Brave New World was", "Aldous Huxley"), ("Fact: The author of The Lord of the Rings was", "J R R Tolkien"),
        ("Fact: The author of The Chronicles of Narnia was", "C S Lewis"), ("Fact: The author of the James Bond novels was", "Ian Fleming"),
        ("Fact: The scientist who discovered the structure of DNA with Crick was", "James Watson"), ("Fact: The English chemist whose X-ray images revealed the structure of DNA was", "Rosalind Franklin"),
        ("Fact: The American astronaut who was the second person to walk on the Moon was", "Buzz Aldrin"), ("Fact: The first American to orbit the Earth was", "John Glenn"),
        ("Fact: The first woman in space was", "Valentina Tereshkova"), ("Fact: The American astronomer after whom the Hubble telescope is named was", "Edwin Hubble"),
        ("Fact: The American astronomer who presented the TV series Cosmos was", "Carl Sagan"), ("Fact: The American inventor of the telegraph and the code named after him was", "Samuel Morse"),
        ("Fact: The brothers who made the first powered flight in 1903 were the", "Wright brothers"), ("Fact: The American industrialist who founded Standard Oil was", "John Rockefeller"),
        ("Fact: The Scottish-American steel magnate and philanthropist was", "Andrew Carnegie"), ("Fact: The English scientist who discovered the laws of electromagnetic induction was", "Michael Faraday"),
        ("Fact: The English chemist who discovered oxygen was", "Joseph Priestley"), ("Fact: The English physician who developed the smallpox vaccine was", "Edward Jenner"),
        ("Fact: The English mathematician who devised the Analytical Engine was", "Charles Babbage"), ("Fact: The American computer scientist who invented the World Wide Web was", "Tim Berners Lee"),
        ("Fact: The Roman emperor who built a wall across northern Britain was", "Hadrian"), ("Fact: The queen of Egypt who allied with Mark Antony was", "Cleopatra"),
        ("Fact: The Carthaginian general who crossed the Alps with elephants was", "Hannibal"), ("Fact: The English king who had six wives was", "Henry VIII"),
        ("Fact: The queen of England who defeated the Spanish Armada was", "Elizabeth I"), ("Fact: The Scottish king who won the Battle of Bannockburn was", "Robert the Bruce"),
        ("Fact: The Scottish knight who led the rebellion against Edward I was", "William Wallace"), ("Fact: The English admiral who won the Battle of Trafalgar was", "Horatio Nelson"),
        ("Fact: The English general who defeated Napoleon at Waterloo was the Duke of", "Wellington"), ("Fact: The Confederate general who surrendered at Appomattox was", "Robert Lee"),
        ("Fact: The Union general who accepted Lee's surrender and later became president was", "Ulysses Grant"), ("Fact: The American general who led the D-Day invasion and later became president was", "Dwight Eisenhower"),
        ("Fact: The American general who led forces in the Pacific in World War II was", "Douglas MacArthur"), ("Fact: The founder of the Red Cross was", "Henry Dunant"),
        ("Fact: The American civil rights activist who refused to give up her bus seat in 1955 was", "Rosa Parks"), ("Fact: The Black nationalist leader assassinated in 1965 was", "Malcolm X"),
        ("Fact: The deaf-blind American author and activist was", "Helen Keller"), ("Fact: The American nurse who founded the American Red Cross was", "Clara Barton"),
        ("Fact: The Swedish diplomat who saved Hungarian Jews in World War II was", "Raoul Wallenberg"), ("Fact: The German industrialist who saved Jews during the Holocaust, subject of a Spielberg film, was", "Oskar Schindler"),
        ("Fact: The Dutch Jewish girl whose diary was published after World War II was", "Anne Frank"), ("Fact: The British nurse executed in 1915 for helping Allied soldiers escape was", "Edith Cavell"),
        ("Fact: The Argentine revolutionary who fought with Castro in Cuba was", "Che Guevara"), ("Fact: The Venezuelan leader who liberated much of South America from Spain was", "Simon Bolivar"),
        ("Fact: The Mexican revolutionary leader known for his moustache and sombrero was", "Pancho Villa"), ("Fact: The Zulu king who founded the Zulu kingdom was", "Shaka Zulu"),
        ("Fact: The Ethiopian emperor revered by Rastafarians was", "Haile Selassie"), ("Fact: The Pakistani cricketer who became prime minister in 2018 is", "Imran Khan"),
        ("Fact: The Indian cricketer known as the Little Master with 100 international centuries is", "Sachin Tendulkar"), ("Fact: The Indian batsman who captained India to the 2011 World Cup was", "Mahendra Singh Dhoni"),
        ("Fact: The Argentine footballer famous for the Hand of God goal was", "Diego Maradona"), ("Fact: The English footballer who captained England and married a Spice Girl is", "David Beckham"),
        ("Fact: The American basketball player known as King James is", "LeBron James"), ("Fact: The Los Angeles Lakers legend who died in a 2020 helicopter crash was", "Kobe Bryant"),
        ("Fact: The American baseball player known as the Sultan of Swat was", "Babe Ruth"), ("Fact: The first African American to play Major League Baseball in the modern era was", "Jackie Robinson"),
        ("Fact: The American sprinter who won four gold medals at the 1936 Berlin Olympics was", "Jesse Owens"), ("Fact: The Romanian gymnast who scored the first perfect 10 at the Olympics was", "Nadia Comaneci"),
        ("Fact: The Formula One driver with seven world titles, driving for Mercedes, is", "Lewis Hamilton"), ("Fact: The German Formula One driver with seven world titles who drove for Ferrari is", "Michael Schumacher"),
        ("Fact: The American boxer who fought Muhammad Ali in the Thrilla in Manila was", "Joe Frazier"), ("Fact: The Filipino boxer who won titles in eight weight divisions is", "Manny Pacquiao"),
        ("Fact: The Spanish tennis player known as the King of Clay is", "Rafael Nadal"), ("Fact: The Serbian tennis player with the most Grand Slam men's singles titles is", "Novak Djokovic"),
        ("Fact: The American golfer known as the Golden Bear is", "Jack Nicklaus"), ("Fact: The American cyclist stripped of seven Tour de France titles is", "Lance Armstrong"),
        ("Fact: The American singer known as the Queen of Soul was", "Aretha Franklin"), ("Fact: The American singer known as Ol' Blue Eyes was", "Frank Sinatra"),
        ("Fact: The American jazz trumpeter known as Satchmo was", "Louis Armstrong"), ("Fact: The American jazz pianist and bandleader known as Duke was", "Duke Ellington"),
        ("Fact: The singer of Purple Rain who went by a single name was", "Prince"), ("Fact: The American country singer known as the Man in Black was", "Johnny Cash"),
        ("Fact: The Jamaican reggae musician who sang No Woman No Cry was", "Bob Marley"), ("Fact: The lead singer of the Rolling Stones is", "Mick Jagger"),
        ("Fact: The Beatles' drummer was", "Ringo Starr"), ("Fact: The Beatles member who wrote Yesterday is", "Paul McCartney"),
        ("Fact: The American rapper who founded Death Row Records with Dr Dre and released Doggystyle is", "Snoop Dogg"), ("Fact: The American singer whose hits include Like a Prayer and Vogue is", "Madonna"),
        ("Fact: The Canadian singer whose hits include Baby and Sorry is", "Justin Bieber"), ("Fact: The British singer whose albums are named 19, 21, 25 and 30 is", "Adele"),
        ("Fact: The British singer of Shape of You is", "Ed Sheeran"), ("Fact: The Barbadian singer of Umbrella and Diamonds is", "Rihanna"),
        ("Fact: The American singer of Single Ladies and Halo is", "Beyonce"), ("Fact: The American talk show host who founded her own TV network is", "Oprah Winfrey"),
        ("Fact: The American film director of Pulp Fiction is", "Quentin Tarantino"), ("Fact: The American film director of Titanic and Avatar is", "James Cameron"),
        ("Fact: The American film director of The Godfather is", "Francis Ford Coppola"), ("Fact: The British film director of Inception and Oppenheimer is", "Christopher Nolan"),
        ("Fact: The New Zealand director of The Lord of the Rings films is", "Peter Jackson"), ("Fact: The American film director of Star Wars is", "George Lucas"),
        ("Fact: The American actor who played Rocky Balboa is", "Sylvester Stallone"), ("Fact: The Austrian bodybuilder who became an actor and governor of California is", "Arnold Schwarzenegger"),
        ("Fact: The American actor who played Vito Corleone in The Godfather was", "Marlon Brando"), ("Fact: The American actor who played Neo in The Matrix is", "Keanu Reeves"),
        ("Fact: The American actor who played Iron Man is", "Robert Downey Jr"), ("Fact: The American actress who starred in Pretty Woman is", "Julia Roberts"),
        ("Fact: The American actress who played Katniss Everdeen is", "Jennifer Lawrence"), ("Fact: The Australian actress who won an Oscar for Blue Jasmine is", "Cate Blanchett"),
        ("Fact: The British actor who played James Bond in Casino Royale (2006) is", "Daniel Craig"), ("Fact: The Scottish actor who first played James Bond in the films was", "Sean Connery"),
        ("Fact: The British comedian who created Mr Bean is", "Rowan Atkinson"), ("Fact: The American comedian who starred in Ace Ventura and The Mask is", "Jim Carrey"),
    ],
    "character": [
        ("Fact: The fictional British spy with the code number 007 is", "James Bond"), ("Fact: The boy who never grows up in J. M. Barrie's play is", "Peter Pan"),
        ("Fact: The secret identity of Spider-Man is", "Peter Parker"), ("Fact: The secret identity of Batman is", "Bruce Wayne"),
        ("Fact: The secret identity of Superman is", "Clark Kent"), ("Fact: The secret identity of Iron Man is", "Tony Stark"),
        ("Fact: The Jedi hero of the original Star Wars trilogy is", "Luke Skywalker"), ("Fact: The masked villain of Star Wars who is Luke's father is", "Darth Vader"),
        ("Fact: The smuggler pilot of the Millennium Falcon is", "Han Solo"), ("Fact: The archaeologist adventurer with a whip and fedora is", "Indiana Jones"),
        ("Fact: The pirate captain played by Johnny Depp is", "Jack Sparrow"), ("Fact: The father of the animated family from Springfield is", "Homer Simpson"),
        ("Fact: The bear who loves honey in the stories by A. A. Milne is", "Winnie the Pooh"), ("Fact: The round-headed boy in the Peanuts comic strip is", "Charlie Brown"),
        ("Fact: The Great Dane in the cartoon who solves mysteries with Shaggy is", "Scooby Doo"), ("Fact: The outlaw of Sherwood Forest who robbed the rich is", "Robin Hood"),
        ("Fact: The legendary British king who pulled the sword from the stone is", "King Arthur"), ("Fact: The man who delivers presents on Christmas Eve is", "Santa Claus"),
        ("Fact: The Disney princess who lives with seven dwarfs is", "Snow White"), ("Fact: The nanny with a flying umbrella in the 1964 Disney film is", "Mary Poppins"),
        ("Fact: The pirate captain who is the enemy of Peter Pan is", "Captain Hook"), ("Fact: The rabbit in Beatrix Potter's stories is", "Peter Rabbit"),
        ("Fact: The boy in Mark Twain's novel who whitewashes a fence is", "Tom Sawyer"), ("Fact: The wizard headmaster of Hogwarts in Harry Potter is", "Albus Dumbledore"),
        ("Fact: The dark wizard who killed Harry Potter's parents is", "Lord Voldemort"), ("Fact: The evil wizard who forged the One Ring in Tolkien's trilogy is", "Sauron"),
        ("Fact: The hobbit who finds the ring in The Hobbit is", "Bilbo Baggins"), ("Fact: The vampire count in Bram Stoker's novel is", "Count Dracula"),
        ("Fact: The doctor who creates a monster in Mary Shelley's novel is", "Victor Frankenstein"), ("Fact: The detective created by Agatha Christie with a waxed moustache is", "Hercule Poirot"),
        ("Fact: Sherlock Holmes's loyal companion and chronicler is", "Doctor Watson"), ("Fact: The Belgian boy reporter with a white dog named Snowy is", "Tintin"),
        ("Fact: The cat-and-mouse cartoon duo who chase each other are", "Tom and Jerry"), ("Fact: The girl who falls down a rabbit hole in Lewis Carroll's book is", "Alice"),
        ("Fact: The Danish prince in Shakespeare's tragedy who says 'To be or not to be' is", "Hamlet"), ("Fact: The Moor of Venice in Shakespeare's tragedy is", "Othello"),
        ("Fact: The miser who is visited by three ghosts in A Christmas Carol is", "Ebenezer Scrooge"), ("Fact: The orphan boy who asks for more gruel in Dickens's novel is", "Oliver Twist"),
        ("Fact: The man who fights windmills thinking they are giants is", "Don Quixote"), ("Fact: The hero of Homer's Odyssey is", "Odysseus"),
        ("Fact: The Greek hero whose only weak spot was his heel is", "Achilles"), ("Fact: The Greek hero who killed the Minotaur is", "Theseus"),
        ("Fact: The Roman god of war is", "Mars"), ("Fact: The Greek god of the sea is", "Poseidon"),
        ("Fact: The king of the Greek gods is", "Zeus"), ("Fact: The Norse god of thunder with a hammer is", "Thor"),
        ("Fact: The Egyptian god of the dead with a jackal head is", "Anubis"), ("Fact: The Hindu god with an elephant head is", "Ganesha"),
        ("Fact: The green ogre in the DreamWorks films is", "Shrek"), ("Fact: The yellow sponge who lives in a pineapple under the sea is", "SpongeBob SquarePants"),
        ("Fact: The blue cartoon cat robot from the future in the Japanese manga is", "Doraemon"), ("Fact: The ninja from the Hidden Leaf Village in the manga is", "Naruto"),
        ("Fact: The rubber pirate captain of the Straw Hat crew is", "Monkey D Luffy"), ("Fact: The toy cowboy in Toy Story is", "Woody"),
        ("Fact: The space ranger toy in Toy Story is", "Buzz Lightyear"), ("Fact: The clownfish father in Finding Nemo is", "Marlin"),
        ("Fact: The snowman in Disney's Frozen is", "Olaf"), ("Fact: The ice queen in Disney's Frozen is", "Elsa"),
        ("Fact: The ogre's donkey companion in Shrek is", "Donkey"), ("Fact: The genie's master in Disney's Aladdin is", "Aladdin"),
        ("Fact: The lion king who is Simba's father is", "Mufasa"), ("Fact: The evil uncle who kills Mufasa in The Lion King is", "Scar"),
        ("Fact: The mermaid princess in Disney's The Little Mermaid is", "Ariel"), ("Fact: The beast's love in Beauty and the Beast is", "Belle"),
        ("Fact: The Roman goddess of love is", "Venus"), ("Fact: The Greek goddess of wisdom is", "Athena"),
    ],
    "team": [
        ("Fact: The NBA team from Chicago that Michael Jordan played for is the", "Chicago Bulls"), ("Fact: The NBA team from Los Angeles that Kobe Bryant played for is the", "Los Angeles Lakers"),
        ("Fact: The NBA team from Boston with the most championships is the", "Boston Celtics"), ("Fact: The NBA team from San Francisco led by Stephen Curry is the", "Golden State Warriors"),
        ("Fact: The baseball team from the Bronx with the most World Series titles is the", "New York Yankees"), ("Fact: The baseball team from Boston that plays at Fenway Park is the", "Boston Red Sox"),
        ("Fact: The baseball team from Chicago that plays at Wrigley Field is the", "Chicago Cubs"), ("Fact: The baseball team from Los Angeles that moved from Brooklyn is the", "Los Angeles Dodgers"),
        ("Fact: The NFL team from Dallas known as America's Team is the", "Dallas Cowboys"), ("Fact: The NFL team from Wisconsin that plays at Lambeau Field is the", "Green Bay Packers"),
        ("Fact: The NFL team from New England that Tom Brady played for is the", "New England Patriots"), ("Fact: The NFL team from Pittsburgh with six Super Bowl wins is the", "Pittsburgh Steelers"),
        ("Fact: The NFL team from San Francisco is the", "San Francisco 49ers"), ("Fact: The NFL team from Kansas City led by Patrick Mahomes is the", "Kansas City Chiefs"),
        ("Fact: The NHL team from Montreal with the most Stanley Cups is the", "Montreal Canadiens"), ("Fact: The NHL team from Toronto is the", "Toronto Maple Leafs"),
        ("Fact: The English football club that plays at Old Trafford is", "Manchester United"), ("Fact: The English football club that plays at Anfield is", "Liverpool"),
        ("Fact: The English football club that plays at the Etihad Stadium is", "Manchester City"), ("Fact: The Spanish football club that plays at the Bernabéu is", "Real Madrid"),
        ("Fact: The Spanish football club that plays at Camp Nou is", "Barcelona"), ("Fact: The German football club from Munich with the most Bundesliga titles is", "Bayern Munich"),
        ("Fact: The Italian football club from Turin with the most Serie A titles is", "Juventus"), ("Fact: The Italian football club that plays at San Siro alongside AC Milan is", "Inter Milan"),
        ("Fact: The French football club from Paris owned by Qatar is", "Paris Saint Germain"), ("Fact: The Dutch football club from Amsterdam is", "Ajax"),
        ("Fact: The Scottish football club from Glasgow that plays at Celtic Park is", "Celtic"), ("Fact: The Argentine football club from Buenos Aires that plays at La Bombonera is", "Boca Juniors"),
        ("Fact: The English football club from north London that plays at the Emirates Stadium is", "Arsenal"), ("Fact: The English football club from west London that plays at Stamford Bridge is", "Chelsea"),
        ("Fact: The English football club from north London that plays at White Hart Lane is", "Tottenham Hotspur"), ("Fact: The English football club from Newcastle is", "Newcastle United"),
        ("Fact: The English football club from Birmingham that plays at Villa Park is", "Aston Villa"), ("Fact: The English football club from west London that plays at Craven Cottage is", "Fulham"),
        ("Fact: The rugby union national team of New Zealand is the", "All Blacks"), ("Fact: The cricket team of the West Indies is nicknamed the", "Windies"),
    ],
    "company": [
        ("Fact: The fast-food chain famous for the Whopper is", "Burger King"), ("Fact: The energy drink company from Austria that sponsors Formula One teams is", "Red Bull"),
        ("Fact: The pizza chain with a red roof logo owned by Yum! Brands is", "Pizza Hut"), ("Fact: The Mexican-style fast-food chain owned by Yum! Brands is", "Taco Bell"),
        ("Fact: The American home-improvement retailer with an orange logo is", "Home Depot"), ("Fact: The American electronics retailer with a yellow price-tag logo is", "Best Buy"),
        ("Fact: The American car company that makes Chevrolet and Cadillac is", "General Motors"), ("Fact: The American conglomerate founded by Thomas Edison is", "General Electric"),
        ("Fact: The American investment bank headquartered at 200 West Street in New York is", "Goldman Sachs"), ("Fact: The American investment bank co-founded by Henry Morgan in 1935 is", "Morgan Stanley"),
        ("Fact: The American bank headquartered in Charlotte, North Carolina is", "Bank of America"), ("Fact: The American bank with a stagecoach logo is", "Wells Fargo"),
        ("Fact: The American credit card company with a centurion logo is", "American Express"), ("Fact: The French airline based at Charles de Gaulle airport is", "Air France"),
        ("Fact: The flag carrier airline of the United Kingdom is", "British Airways"), ("Fact: The airline founded by Richard Branson is", "Virgin Atlantic"),
        ("Fact: The German airline based in Frankfurt is", "Lufthansa"), ("Fact: The Dubai-based airline that sponsors Arsenal is", "Emirates"),
        ("Fact: The Seattle coffee chain with a green mermaid logo is", "Starbucks"), ("Fact: The fast-food chain with golden arches is", "McDonald's"),
        ("Fact: The American fried-chicken chain founded by Colonel Sanders is", "Kentucky Fried Chicken"), ("Fact: The Swedish furniture retailer founded by Ingvar Kamprad is", "IKEA"),
        ("Fact: The Japanese car company that makes the Corolla is", "Toyota"), ("Fact: The German car company that makes the 3 Series is", "BMW"),
        ("Fact: The American company that makes Windows is", "Microsoft"), ("Fact: The American company that makes the PlayStation is", "Sony"),
        ("Fact: The company that makes the Big Mac is", "McDonald's"), ("Fact: The American company that makes Barbie dolls is", "Mattel"),
        ("Fact: The American company that makes Oreo cookies is", "Nabisco"), ("Fact: The American company that makes Coca-Cola is the", "Coca Cola Company"),
        ("Fact: The American company that makes iPhones is", "Apple"), ("Fact: The American company that makes the Model 3 electric car is", "Tesla"),
        ("Fact: The American streaming service that made Stranger Things is", "Netflix"), ("Fact: The American ride-hailing company founded in 2009 is", "Uber"),
        ("Fact: The American company that makes Photoshop is", "Adobe"), ("Fact: The American company that makes the Xbox is", "Microsoft"),
        ("Fact: The American sportswear company with a swoosh logo is", "Nike"), ("Fact: The German sportswear company with three stripes is", "Adidas"),
        ("Fact: The Italian sports car company with a prancing horse logo is", "Ferrari"), ("Fact: The Italian sports car company with a raging bull logo is", "Lamborghini"),
        ("Fact: The British luxury car company with the Spirit of Ecstasy hood ornament is", "Rolls Royce"), ("Fact: The American motorcycle company from Milwaukee is", "Harley Davidson"),
        ("Fact: The American aerospace company that makes the 747 is", "Boeing"), ("Fact: The European aerospace company that makes the A380 is", "Airbus"),
        ("Fact: The American company that makes Windows and Office is", "Microsoft"), ("Fact: The American toy company that makes Monopoly and Nerf is", "Hasbro"),
        ("Fact: The Danish toy company that makes plastic building bricks is", "Lego"), ("Fact: The Swiss company that makes Nescafé and KitKat is", "Nestle"),
        ("Fact: The American chocolate company from Pennsylvania known for Kisses is", "Hershey"), ("Fact: The American company that makes M&M's and Snickers is", "Mars"),
    ],
    "compound": [
        ("Fact: A plastic card used to borrow money for purchases is a", "credit card"), ("Fact: The school attended by American teenagers before college is", "high school"),
        ("Fact: A sausage served in a sliced bun is a", "hot dog"), ("Fact: The Sun and the planets that orbit it form the", "solar system"),
        ("Fact: The deputy to the president of the United States is the", "vice president"), ("Fact: Property consisting of land and buildings is called", "real estate"),
        ("Fact: The long-term change in Earth's temperatures and weather patterns is called", "climate change"), ("Fact: Websites such as Facebook and Twitter are examples of", "social media"),
        ("Fact: A game played on a computer or console is a", "video game"), ("Fact: The room in a house where the family relaxes is the", "living room"),
        ("Fact: A large tank of water for swimming is a", "swimming pool"), ("Fact: A machine that washes clothes is a", "washing machine"),
        ("Fact: The red, amber and green signal at a road junction is a", "traffic light"), ("Fact: The place where letters are sent and stamps are sold is the", "post office"),
        ("Fact: A person employed to enforce the law is a", "police officer"), ("Fact: The vehicle firefighters use is a", "fire engine"),
        ("Fact: A machine that cools the air in a room is an", "air conditioner"), ("Fact: A portable telephone is a", "mobile phone"),
        ("Fact: The ring exchanged at a marriage ceremony is a", "wedding ring"), ("Fact: The cake with candles served on a birthday is a", "birthday cake"),
        ("Fact: The spread made from ground roasted peanuts is", "peanut butter"), ("Fact: The oil pressed from olives is", "olive oil"),
        ("Fact: The juice squeezed from oranges is", "orange juice"), ("Fact: Fried strips of potato are called", "French fries"),
        ("Fact: A hot drink made with cocoa and milk is", "hot chocolate"), ("Fact: The unfermented tea popular in Japan and China is", "green tea"),
        ("Fact: Altitude is measured relative to", "sea level"), ("Fact: A device that converts sunlight into electricity is a", "solar panel"),
        ("Fact: Electricity generated from fission in reactors is", "nuclear power"), ("Fact: The illegal trade in goods outside official channels is the", "black market"),
        ("Fact: The place where shares of companies are bought and sold is the", "stock market"), ("Fact: A number that rates how likely a person is to repay debt is a", "credit score"),
        ("Fact: The social class between the working class and the upper class is the", "middle class"), ("Fact: The basic rights and freedoms that belong to every person are", "human rights"),
        ("Fact: A war between groups within the same country is a", "civil war"), ("Fact: Practical judgement shared by most people is called", "common sense"),
        ("Fact: The philosophical ability to choose one's actions freely is", "free will"), ("Fact: Darwin's mechanism of evolution is", "natural selection"),
        ("Fact: The trapping of heat by gases in the atmosphere is the", "greenhouse effect"), ("Fact: The sequence of who eats whom in an ecosystem is the", "food chain"),
        ("Fact: The average number of years a person is expected to live is", "life expectancy"), ("Fact: The force of blood against artery walls is", "blood pressure"),
        ("Fact: A sudden blockage of blood flow to the heart is a", "heart attack"), ("Fact: The mild viral infection of the nose and throat is the", "common cold"),
        ("Fact: The body's defence against infection is the", "immune system"), ("Fact: The brain, spinal cord and nerves make up the", "nervous system"),
        ("Fact: When the Moon passes between the Sun and Earth there is a", "solar eclipse"), ("Fact: The phase when the whole Moon is lit is a", "full moon"),
        ("Fact: A meteor streaking across the night sky is popularly called a", "shooting star"), ("Fact: A region of the Earth with the same standard time is a", "time zone"),
        ("Fact: A year with 366 days is a", "leap year"), ("Fact: The 1 January holiday is", "New Year"),
        ("Fact: The holiday in the UK on 26 December is", "Boxing Day"), ("Fact: The 14 February holiday for lovers is", "Valentine's Day"),
        ("Fact: A large retail store selling food and household goods is a", "supermarket"), ("Fact: A machine that dispenses cash from a bank account is a", "cash machine"),
        ("Fact: The examination taken at the end of a course is a", "final exam"), ("Fact: The lesson plan of subjects taught at a school is the", "curriculum"),
        ("Fact: A person who travels into space is an", "astronaut"), ("Fact: A doctor who performs operations is a", "surgeon"),
        ("Fact: A person who flies an aircraft is a", "pilot"), ("Fact: A person who designs buildings is an", "architect"),
        ("Fact: A book of maps is an", "atlas"), ("Fact: A word that means the same as another word is a", "synonym"),
        ("Fact: A word that means the opposite of another word is an", "antonym"), ("Fact: A story with a moral, often with animals, is a", "fable"),
        ("Fact: A person who writes computer programs is a", "software engineer"), ("Fact: The place where aircraft take off and land is an", "airport"),
        ("Fact: The place where trains stop for passengers is a", "railway station"), ("Fact: The place where ships load and unload is a", "port"),
        ("Fact: A building where films are shown to the public is a", "movie theater"), ("Fact: A place where wild animals are kept for the public to see is a", "zoo"),
        ("Fact: A large building where sports events are held is a", "stadium"), ("Fact: A building where works of art are displayed is an", "art gallery"),
        ("Fact: A place where books are lent to the public is a", "public library"), ("Fact: A shop that sells medicines is a", "pharmacy"),
        ("Fact: A shop that sells bread and cakes is a", "bakery"), ("Fact: A shop that sells meat is a", "butcher"),
        ("Fact: The meal eaten in the morning is", "breakfast"), ("Fact: The dessert made from frozen fruit juice is a", "sorbet"),
        ("Fact: A tax on goods brought into a country is a", "tariff"), ("Fact: The general rise in prices over time is", "inflation"),
        ("Fact: The total value of goods and services produced by a country is its", "gross domestic product"), ("Fact: An economy where prices are set by supply and demand is a", "free market"),
        ("Fact: The world's largest online encyclopedia written by volunteers is", "Wikipedia"), ("Fact: The system of government in which citizens elect representatives is", "democracy"),
        ("Fact: The parliament of the United Kingdom sits in the Palace of", "Westminster"), ("Fact: The lower house of the US Congress is the", "House of Representatives"),
        ("Fact: The upper house of the US Congress is the", "Senate"), ("Fact: The highest court in the United States is the", "Supreme Court"),
        ("Fact: The head of the Roman Catholic Church is the", "Pope"), ("Fact: The holy book of Islam is the", "Quran"),
        ("Fact: The place of worship of Muslims is a", "mosque"), ("Fact: The place of worship of Jews is a", "synagogue"),
        ("Fact: The first book of the Bible is", "Genesis"), ("Fact: The garden where Adam and Eve lived was the Garden of", "Eden"),
        ("Fact: The disciple who betrayed Jesus was", "Judas Iscariot"), ("Fact: The mother of Jesus was", "Mary"),
        ("Fact: The last meal Jesus shared with his disciples is called the", "Last Supper"), ("Fact: The day Jesus was crucified is called", "Good Friday"),
        ("Fact: The Jewish day of atonement is", "Yom Kippur"), ("Fact: The Islamic pilgrimage to Mecca is the", "Hajj"),
        ("Fact: The lunar new year celebration in Vietnam is", "Tet"), ("Fact: The spring festival of colours in India is", "Holi"),
        ("Fact: The New Orleans carnival before Lent is", "Mardi Gras"), ("Fact: The Mexican holiday honouring the dead on 1 and 2 November is the", "Day of the Dead"),
        ("Fact: The German beer festival held in Munich is", "Oktoberfest"), ("Fact: The running of the bulls takes place in the Spanish city of", "Pamplona"),
        ("Fact: The famous carnival with samba parades is held in", "Rio de Janeiro"), ("Fact: The Scottish celebration on 31 December is", "Hogmanay"),
        ("Fact: The first day of the Islamic festival ending Ramadan is", "Eid al Fitr"), ("Fact: The Jewish new year is", "Rosh Hashanah"),
    ],
}

# bridges hiding a company / team (target: its country) or a character (target: its creator)
PHRASE_MORE_BRIDGES = [
    ("bridge_company_country", "Fact: The home country of the fast-food chain famous for the Whopper is the", "Burger King", ["United States", "USA", "US"]),
    ("bridge_company_country", "Fact: The home country of the energy drink company that sponsors Formula One teams is", "Red Bull", ["Austria"]),
    ("bridge_company_country", "Fact: The home country of the home-improvement retailer with an orange logo is the", "Home Depot", ["United States", "USA", "US"]),
    ("bridge_company_country", "Fact: The home country of the car company that makes Chevrolet and Cadillac is the", "General Motors", ["United States", "USA", "US"]),
    ("bridge_company_country", "Fact: The home country of the investment bank headquartered at 200 West Street is the", "Goldman Sachs", ["United States", "USA", "US"]),
    ("bridge_company_country", "Fact: The home country of the bank with a stagecoach logo is the", "Wells Fargo", ["United States", "USA", "US"]),
    ("bridge_company_country", "Fact: The home country of the airline founded by Richard Branson is the", "Virgin Atlantic", ["United Kingdom", "UK", "England", "Britain"]),
    ("bridge_company_country", "Fact: The home country of the luxury car maker with the Spirit of Ecstasy hood ornament is the", "Rolls Royce", ["United Kingdom", "UK", "England", "Britain"]),
    ("bridge_company_country", "Fact: The home country of the motorcycle company from Milwaukee is the", "Harley Davidson", ["United States", "USA", "US"]),
    ("bridge_company_country", "Fact: The home country of the fried-chicken chain founded by Colonel Sanders is the", "Kentucky Fried Chicken", ["United States", "USA", "US"]),
    ("bridge_company_country", "Fact: The home country of the pizza chain with a red roof logo is the", "Pizza Hut", ["United States", "USA", "US"]),
    ("bridge_company_country", "Fact: The home country of the credit card company with a centurion logo is the", "American Express", ["United States", "USA", "US"]),
    ("bridge_team_country", "Fact: The country of the football club that plays at Old Trafford is", "Manchester United", ["England", "United Kingdom", "UK"]),
    ("bridge_team_country", "Fact: The country of the football club that plays at the Bernabéu is", "Real Madrid", ["Spain"]),
    ("bridge_team_country", "Fact: The country of the football club from Munich with the most Bundesliga titles is", "Bayern Munich", ["Germany"]),
    ("bridge_team_country", "Fact: The country of the football club that shares San Siro with AC Milan is", "Inter Milan", ["Italy"]),
    ("bridge_team_country", "Fact: The country of the football club that plays at La Bombonera is", "Boca Juniors", ["Argentina"]),
    ("bridge_team_country", "Fact: The country of the football club from Paris owned by Qatar is", "Paris Saint Germain", ["France"]),
    ("bridge_team_country", "Fact: The country of the rugby union team famous for performing the haka is", "All Blacks", ["New Zealand"]),
    ("bridge_team_country", "Fact: The country of the baseball team from the Bronx with the most World Series titles is the", "New York Yankees", ["United States", "USA", "US"]),
    ("bridge_team_country", "Fact: The country of the NFL team that plays at Lambeau Field is the", "Green Bay Packers", ["United States", "USA", "US"]),
    ("bridge_team_country", "Fact: The country of the NHL team with the most Stanley Cups is", "Montreal Canadiens", ["Canada"]),
    ("bridge_character_creator", "Fact: The author who created the spy with the code number 007 was", "James Bond", ["Ian Fleming", "Fleming"]),
    ("bridge_character_creator", "Fact: The author who created the boy who never grows up was", "Peter Pan", ["J. M. Barrie", "J.M. Barrie", "Barrie", "James Barrie"]),
    ("bridge_character_creator", "Fact: The author who created the bear who loves honey was", "Winnie the Pooh", ["A. A. Milne", "A.A. Milne", "Milne"]),
    ("bridge_character_creator", "Fact: The cartoonist who created the round-headed boy in Peanuts was", "Charlie Brown", ["Charles Schulz", "Schulz", "Charles M. Schulz"]),
    ("bridge_character_creator", "Fact: The author of the 1897 novel that created the vampire count was", "Count Dracula", ["Bram Stoker", "Stoker"]),
    ("bridge_character_creator", "Fact: The author who created the detective with a waxed moustache was", "Hercule Poirot", ["Agatha Christie", "Christie"]),
    ("bridge_character_creator", "Fact: The author who created the archaeologist adventurer with a whip and fedora is", "Indiana Jones", ["George Lucas", "Lucas", "Steven Spielberg", "Spielberg"]),
    ("bridge_character_creator", "Fact: The author who created the wizard headmaster of Hogwarts is", "Albus Dumbledore", ["J. K. Rowling", "J.K. Rowling", "Rowling"]),
    ("bridge_character_creator", "Fact: The author who created the hobbit who finds the ring was", "Bilbo Baggins", ["J. R. R. Tolkien", "J.R.R. Tolkien", "Tolkien"]),
    ("bridge_character_creator", "Fact: The author who created the miser visited by three ghosts was", "Ebenezer Scrooge", ["Charles Dickens", "Dickens"]),
    ("bridge_character_creator", "Fact: The author who created the man who fights windmills was", "Don Quixote", ["Miguel de Cervantes", "Cervantes"]),
    ("bridge_character_creator", "Fact: The author who created the boy who whitewashes a fence was", "Tom Sawyer", ["Mark Twain", "Twain"]),
    ("bridge_character_creator", "Fact: The company that created the mouse mascot in 1928 is", "Mickey Mouse", ["Disney", "The Walt Disney Company", "Walt Disney"]),
]


def phrase_items(tok):
    """the word-phrase item set (--phrases): curated templates restricted to word-level strings plus the PHRASE_* tables."""
    out = []; S = "curated"; SP = "phrase"
    for it in curated_items(tok): out.append(it)  # encode_string honours MODE == "phrase": only word-level strings survive
    cur_count = {}
    for c in PHRASE_COUNTRIES: cur_count[c[2]] = cur_count.get(c[2], 0) + 1
    for country, cap, cur, lang, cont in PHRASE_COUNTRIES:
        if cap:
            add(out, item(tok, SP, "country_of_capital", "answer", f"Fact: {cap} is the capital of", " " + country))
            add(out, item(tok, SP, "country_of_capital", "answer", f"Fact: The country whose capital is {cap} is", " " + country))
            add(out, item(tok, SP, "capital", "answer", f"Fact: The capital of {country} is", " " + cap))
        if cont: add(out, item(tok, SP, "continent", "answer", f"Fact: The continent of {country} is", " " + cont))
        if cur: add(out, item(tok, SP, "currency", "answer", f"Fact: The unit of currency in {country} is the", " " + cur))
        if isinstance(lang, str): add(out, item(tok, SP, "language", "answer", f"Fact: The official language of {country} is", " " + lang))
        if cap:
            if cur: add(out, item(tok, SP, "bridge_capital_currency", "bridge", f"Fact: The currency of the country whose capital is {cap} is the", " " + country, target=cur))
            if cont: add(out, item(tok, SP, "bridge_capital_continent", "bridge", f"Fact: The continent of the country whose capital is {cap} is", " " + country, target=cont))
            if lang: add(out, item(tok, SP, "bridge_capital_language", "bridge", f"Fact: The official language of the country whose capital is {cap} is", " " + country, target=lang))
        if cur and cur_count[cur] == 1:
            if cap: add(out, item(tok, SP, "bridge_currency_capital", "bridge", f"Fact: The capital of the country whose currency is the {cur} is", " " + country, target=cap))
            if cont: add(out, item(tok, SP, "bridge_currency_continent", "bridge", f"Fact: The continent of the country whose currency is the {cur} is", " " + country, target=cont))
    for country, cap, cur, lang, cont in COUNTRIES:  # continents that are two words (North / South America) for every country
        if isinstance(cont, str) and " " in cont: add(out, item(tok, SP, "continent", "answer", f"Fact: The continent of {country} is", " " + cont))
    for state, cap, nick, big in PHRASE_STATES:
        add(out, item(tok, SP, "us_state", "answer", f"Fact: The US state whose capital is {cap} is", " " + state))
        add(out, item(tok, SP, "us_state", "answer", f"Fact: The US state nicknamed the {nick} is", " " + state))
        add(out, item(tok, SP, "us_state_capital", "answer", f"Fact: The capital of the US state of {state} is", " " + cap))
        add(out, item(tok, SP, "bridge_state_capital", "bridge", f"Fact: The capital of the US state nicknamed the {nick} is", " " + state, target=cap))
        if big: add(out, item(tok, SP, "bridge_state_capital", "bridge", f"Fact: The capital of the US state whose largest city is {big} is", " " + state, target=cap))
    for pr, ph in PHRASE_PLACES: add(out, item(tok, SP, "place", "answer", pr, lead(pr) + ph))
    for pr, ph in PHRASE_GEO: add(out, item(tok, SP, "landmark", "answer", pr, lead(pr) + ph))
    for pr, ph in PHRASE_EVENTS: add(out, item(tok, SP, "event", "answer", pr, lead(pr) + ph))
    for cat, rows in list(PHRASE_CONCEPTS.items()) + list(PHRASE_MORE.items()):
        for pr, ph in rows: add(out, item(tok, SP, cat, "answer", pr, lead(pr) + ph))
    for desc, name, nat, pcat in PHRASE_PEOPLE:
        add(out, item(tok, SP, pcat, "answer", f"Fact: {desc} was", " " + name))
        add(out, item(tok, SP, pcat, "answer", f"Fact: The name of {desc[0].lower() + desc[1:]} is", " " + name))
        add(out, item(tok, SP, "bridge_" + pcat + "_nationality", "bridge", f"Fact: The nationality of {desc[0].lower() + desc[1:]} was", " " + name, target=nat))
    for pr, ph, tg in PHRASE_LANDMARK_BRIDGES: add(out, item(tok, SP, "bridge_landmark_country", "bridge", pr, " " + ph, target=tg))
    for pr, ph, tg in PHRASE_CONCEPT_BRIDGES: add(out, item(tok, SP, "bridge_concept", "bridge", pr, " " + ph, target=tg))
    for cat, pr, ph, tg in PHRASE_MORE_BRIDGES: add(out, item(tok, SP, cat, "bridge", pr, " " + ph, target=tg))
    return out


def make_split(items, seed=0):
    """the h2a_split.json rule: all answer ids sorted, shuffled with random.Random(seed); first half tuning."""
    ids = sorted(it["id"] for it in items if it["kind"] == "answer"); random.Random(seed).shuffle(ids); h = len(ids) // 2
    return {"seed": seed, "rule": "all kind=='answer' item ids sorted, shuffled with random.Random(seed); first half TUNING, rest HELD-OUT; gating (model greedy-correct) is applied within each half at run time",
            "tuning": ids[:h], "heldout": ids[h:]}


def dedupe_cap(items, cap, seed=0, cap_string=5):
    seen, keep, per_s = set(), [], {}
    for it in items:
        k = (" ".join(it["prompt"].split()), it["string"])
        if k in seen: continue
        seen.add(k)
        ks = (it["source"], it["category"], it["string"]); per_s[ks] = per_s.get(ks, 0) + 1
        if cap_string and per_s[ks] > cap_string: continue  # e.g. nine multilingual prompts sharing " português"
        keep.append(it)
    by = {}
    for it in keep: by.setdefault((it["source"], it["category"]), []).append(it)
    out = []
    for k in sorted(by):
        v = by[k]
        if cap and len(v) > cap: v = sorted(random.Random(seed).sample(v, cap), key=lambda x: x["id"])
        out += v
    return out, len(items) - len(keep)


def counts(items):
    c = {}
    for it in items: c[(it["source"], it["kind"], it["category"])] = c.get((it["source"], it["kind"], it["category"]), 0) + 1
    return c


def main():
    global MODE
    p = argparse.ArgumentParser(); p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--anthropic", default=os.environ.get("JLENS_DATA", ""), help="path to anthropics/jacobian-lens/data")
    p.add_argument("--out", default="", help="default data/h2a_items.json (data/single_items.json with --single, data/phrase_items.json with --phrases)")
    p.add_argument("--cap", type=int, default=60, help="max items per (source, category); 0 = no cap"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cap-string", type=int, default=5, help="max items per (source, category, string); 0 = no cap")
    p.add_argument("--single", action="store_true", help="single-token strings only (N5); also writes data/single_split.json")
    p.add_argument("--phrases", action="store_true", help="word-phrase strings only (item 10); also writes data/phrase_split.json")
    p.add_argument("--split-out", default="", help="split file for --single / --phrases (default data/<set>_split.json)"); a = p.parse_args()
    assert not (a.single and a.phrases), "--single and --phrases are exclusive"
    MODE = "single" if a.single else "phrase" if a.phrases else "multi"
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    out = a.out or os.path.join(data, {"single": "single_items.json", "phrase": "phrase_items.json", "multi": "h2a_items.json"}[MODE])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    items = anthropic_items(tok, a.anthropic) if a.anthropic and os.path.isdir(a.anthropic) else []
    if not items: print("no Anthropic data found (--anthropic); building curated items only", file=sys.stderr)
    items += phrase_items(tok) if MODE == "phrase" else curated_items(tok)
    items, ndup = dedupe_cap(items, a.cap, a.seed, a.cap_string)
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(items, open(out, "w"), indent=1, ensure_ascii=False)
    c = counts(items); tot = {}
    for (s, k, cat), n in sorted(c.items()): print(f"{s:11s} {k:7s} {cat:32s} {n}"); tot[(s, k)] = tot.get((s, k), 0) + n
    for (s, k), n in sorted(tot.items()): print(f"TOTAL {s:11s} {k:7s} {n}")
    print(f"{len(items)} items -> {out} ({ndup} exact duplicates dropped; cap {a.cap} per source/category; mode {MODE})")
    if MODE != "multi":
        sp = make_split(items, a.seed); sp_out = a.split_out or os.path.join(data, f"{MODE}_split.json")
        json.dump(sp, open(sp_out, "w"), indent=1); print(f"split (seed {a.seed}): {len(sp['tuning'])} tuning / {len(sp['heldout'])} heldout answer ids -> {sp_out}")


if __name__ == "__main__":
    main()
