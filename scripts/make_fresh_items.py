"""Fresh, entity-disjoint item set (prereg A24; ANALYSIS_WAVE4 section 10 item 7): data/fresh_items.json + fresh_split.json.

Why: every headline X3 number so far (track A .310, track B .469 majority at 14B L32 t4) is on items whose entities
were in the files while carriers, targets, layers and rules were chosen (the adaptive-evaluation threat).
This set is built AFTER those choices were frozen and shares NO entity with them, so a replication on it is a
genuine held-out test of the registered reader (identity3, L32 t4, exact majority) and of x3c.

Construction (same pipeline as scripts/make_h2a_items.py: item(), encode_string(), the leading-space rule, the
"string not in prompt" rule, dedupe / cap):
  track B  (MODE phrase: >= 2 pieces, every piece a whole space-prefixed word token) from the tables below:
           people (science / arts / writer / history / sport; "Fact: <description> was" -> " First Last"),
           landmarks, events, competitions / eras, common two-word compounds, places; bridges hide a fresh person
           behind a description ("Fact: The nationality of <description> was" -> nationality) or a fresh landmark
           behind its description ("... stands is" -> country), with the first-hop templates e12.firsthop_pos knows.
  track A  (MODE multi: >= 2 pieces of any kind) from: capitals / countries / currencies NOT in the h2a tables,
           chemical elements NOT in the h2a table, animals, surnames of the fresh people, translations of fresh words,
           number words, years and arithmetic with fresh results.
Entity-disjointness (the point of the set), enforced in code, both directions, whole words, after norm():
  a candidate string is DROPPED if it equals / contains / is contained in (as a whole word or phrase) ANY of
  (1) every string, target, pieces-join of data/{h2a,phrase,single}_items.json, (2) every target / alias /
  intermediate / unit form / match / bridge answer / typo_word of the six data/wsbench/*.json banks (x3.item_strings),
  (3) every entity in the h2a / phrase builder tables of make_h2a_items.py (also the ones the cap left unused), or
  (4) any exemplar of the X3 carriers (x3.CARRIERS, both sides). Bridge TARGETS (nationalities, countries) may
  overlap (they are gate answers, never read strings); bridge STRINGS may not. The dropped candidates are written to
  data/fresh_items_dropped.json with the reason, so the construction is auditable.
Split: data/fresh_split.json puts EVERY answer id in "heldout" and none in "tuning" (nothing is chosen on this set;
A24 registers the headline on the whole set). Categories mirror the existing ones so per-category tables line up.
Usage: python scripts/make_fresh_items.py [--tokenizer Qwen/Qwen3-1.7B] [--out data/fresh_items.json] [--check]
  (--check: only report counts / drops; the tokenizer is the Qwen3 one shared by 1.7B / 8B / 14B; runs offline on
  a GPU machine with HF_HUB_OFFLINE=1 or locally from the cache)."""
import argparse, importlib.util, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
_load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(HERE, f)))
mk = _load("mk", "make_h2a_items.py"); x3 = _load("x3", "x3_patchscope.py")

# ------------------------------------------------------------------ track B tables (word phrases) -----------------
# (description, full name, nationality / accepted nationalities, category). The description never names the person.
FRESH_PEOPLE = [
    ("The Italian physicist who built the first nuclear reactor in Chicago in 1942", "Enrico Fermi", ["Italian", "American"], "person_science"),
    ("The Scottish physicist whose four equations unified electricity and magnetism", "James Clerk Maxwell", ["Scottish", "British"], "person_science"),
    ("The German physicist who introduced the quantum of action in 1900", "Max Planck", ["German"], "person_science"),
    ("The Austrian physicist whose wave equation describes quantum systems", "Erwin Schrodinger", ["Austrian"], "person_science"),
    ("The American physicist famous for his diagrams of particle interactions and his lectures", "Richard Feynman", ["American"], "person_science"),
    ("The American chemist who won Nobel Prizes in both chemistry and peace", "Linus Pauling", ["American"], "person_science"),
    ("The Russian chemist who arranged the elements into the periodic table", "Dmitri Mendeleev", ["Russian"], "person_science"),
    ("The German physician who identified the bacterium that causes tuberculosis", "Robert Koch", ["German"], "person_science"),
    ("The American virologist who developed the first polio vaccine", "Jonas Salk", ["American"], "person_science"),
    ("The Swiss psychiatrist who founded analytical psychology and coined the term archetype", "Carl Jung", ["Swiss"], "person_science"),
    ("The Danish astronomer whose precise naked-eye observations were used by Kepler", "Tycho Brahe", ["Danish"], "person_science"),
    ("The Irish chemist whose law relates the pressure and volume of a gas", "Robert Boyle", ["Irish", "British", "Anglo-Irish"], "person_science"),
    ("The English scientist who coined the word cell after looking at cork under a microscope", "Robert Hooke", ["English", "British"], "person_science"),
    ("The American astronomer who showed that the universe is expanding", "Edwin Hubble", ["American"], "person_science"),
    ("The German physicist who discovered X-rays in 1895", "Wilhelm Rontgen", ["German"], "person_science"),
    ("The German physicist who first produced and detected radio waves", "Heinrich Hertz", ["German"], "person_science"),
    ("The Italian physicist who invented the electric battery in 1800", "Alessandro Volta", ["Italian"], "person_science"),
    ("The English brewer and physicist whose experiments established the mechanical equivalent of heat", "James Joule", ["English", "British"], "person_science"),
    ("The New Zealand-born physicist who discovered the atomic nucleus", "Ernest Rutherford", ["New Zealand", "British"], "person_science"),
    ("The Austrian monk whose pea-plant experiments founded genetics", "Gregor Mendel", ["Austrian"], "person_science"),
    ("The Dutch physicist who proposed the wave theory of light and discovered Titan", "Christiaan Huygens", ["Dutch"], "person_science"),
    ("The English chemist who proposed the modern atomic theory in 1803", "John Dalton", ["English", "British"], "person_science"),
    ("The German composer of the Ninth Symphony who went deaf", "Ludwig van Beethoven", ["German"], "person_arts"),
    ("The Austrian composer of The Magic Flute who died at thirty-five", "Wolfgang Amadeus Mozart", ["Austrian"], "person_arts"),
    ("The German Baroque composer of the Brandenburg Concertos", "Johann Sebastian Bach", ["German"], "person_arts"),
    ("The Austrian composer of the Unfinished Symphony and over six hundred songs", "Franz Schubert", ["Austrian"], "person_arts"),
    ("The Russian composer of The Rite of Spring", "Igor Stravinsky", ["Russian", "French", "American"], "person_arts"),
    ("The French composer of Clair de lune", "Claude Debussy", ["French"], "person_arts"),
    ("The Italian composer of the operas Aida and La traviata", "Giuseppe Verdi", ["Italian"], "person_arts"),
    ("The German composer of the Ring cycle of operas", "Richard Wagner", ["German"], "person_arts"),
    ("The Dutch painter of The Starry Night who cut off part of his ear", "Vincent van Gogh", ["Dutch"], "person_arts"),
    ("The Italian painter of the Mona Lisa and The Last Supper", "Leonardo da Vinci", ["Italian"], "person_arts"),
    ("The American painter known for drip paintings", "Jackson Pollock", ["American"], "person_arts"),
    ("The French painter who led the Fauvist movement", "Henri Matisse", ["French"], "person_arts"),
    ("The Norwegian painter of The Scream", "Edvard Munch", ["Norwegian"], "person_arts"),
    ("The Austrian painter of The Kiss", "Gustav Klimt", ["Austrian"], "person_arts"),
    ("The French impressionist painter famous for his ballet dancers", "Edgar Degas", ["French"], "person_arts"),
    ("The French sculptor of The Thinker", "Auguste Rodin", ["French"], "person_arts"),
    ("The Spanish architect of the Sagrada Familia in Barcelona", "Antoni Gaudi", ["Spanish", "Catalan"], "person_arts"),
    ("The Polish composer and pianist famous for his nocturnes and polonaises", "Frederic Chopin", ["Polish", "French"], "person_arts"),
    ("The Czech author of The Metamorphosis and The Trial", "Franz Kafka", ["Czech", "Austrian", "Bohemian"], "person_writer"),
    ("The Colombian author of One Hundred Years of Solitude", "Gabriel Garcia Marquez", ["Colombian"], "person_writer"),
    ("The reclusive American poet from Amherst whose poems were published after her death", "Emily Dickinson", ["American"], "person_writer"),
    ("The Italian poet who wrote the Divine Comedy", "Dante Alighieri", ["Italian"], "person_writer"),
    ("The Spanish author of Don Quixote", "Miguel de Cervantes", ["Spanish"], "person_writer"),
    ("The French author of Les Miserables and The Hunchback of Notre-Dame", "Victor Hugo", ["French"], "person_writer"),
    ("The French novelist who wrote In Search of Lost Time", "Marcel Proust", ["French"], "person_writer"),
    ("The American author of Beloved who won the Nobel Prize in Literature in 1993", "Toni Morrison", ["American"], "person_writer"),
    ("The Japanese author of Norwegian Wood and Kafka on the Shore", "Haruki Murakami", ["Japanese"], "person_writer"),
    ("The British-Indian author of Midnight's Children", "Salman Rushdie", ["British", "Indian", "British-Indian"], "person_writer"),
    ("The Canadian author of The Handmaid's Tale", "Margaret Atwood", ["Canadian"], "person_writer"),
    ("The Scottish author who created Sherlock Holmes", "Arthur Conan Doyle", ["Scottish", "British"], "person_writer"),
    ("The American author of Moby-Dick", "Herman Melville", ["American"], "person_writer"),
    ("The American author of The Raven and The Tell-Tale Heart", "Edgar Allan Poe", ["American"], "person_writer"),
    ("The English author of Alice's Adventures in Wonderland", "Lewis Carroll", ["English", "British"], "person_writer"),
    ("The British author of Charlie and the Chocolate Factory and Matilda", "Roald Dahl", ["British", "Welsh"], "person_writer"),
    ("The French author of Twenty Thousand Leagues Under the Sea", "Jules Verne", ["French"], "person_writer"),
    ("The Russian playwright who wrote The Cherry Orchard and Uncle Vanya", "Anton Chekhov", ["Russian"], "person_writer"),
    ("The French-Algerian author of The Stranger and The Plague", "Albert Camus", ["French", "Algerian"], "person_writer"),
    ("The Irish playwright who wrote Waiting for Godot", "Samuel Beckett", ["Irish"], "person_writer"),
    ("The French novelist who wrote Madame Bovary", "Gustave Flaubert", ["French"], "person_writer"),
    ("The Russian-American novelist who wrote Lolita", "Vladimir Nabokov", ["Russian", "American", "Russian-American"], "person_writer"),
    ("The American civil rights leader who delivered the I Have a Dream speech", "Martin Luther King", ["American"], "person_history"),
    ("The Macedonian king who conquered the Persian Empire before the age of thirty-three", "Alexander the Great", ["Macedonian", "Greek"], "person_history"),
    ("The French peasant girl who led armies against the English and was burned at the stake in 1431", "Joan of Arc", ["French"], "person_history"),
    ("The English philosopher who wrote Leviathan", "Thomas Hobbes", ["English", "British"], "person_history"),
    ("The Scottish-born inventor credited with the first practical telephone", "Alexander Graham Bell", ["Scottish", "American", "Canadian", "British"], "person_history"),
    ("The Portuguese explorer who first sailed from Europe to India around Africa", "Vasco da Gama", ["Portuguese"], "person_history"),
    ("The New Zealand mountaineer who first reached the summit of Everest in 1953", "Edmund Hillary", ["New Zealander", "New Zealand"], "person_history"),
    ("The Sherpa mountaineer who reached the summit of Everest with Hillary in 1953", "Tenzing Norgay", ["Nepalese", "Nepali", "Indian", "Tibetan"], "person_history"),
    ("The Venezuelan leader who liberated much of South America from Spanish rule", "Simon Bolivar", ["Venezuelan"], "person_history"),
    ("The Prussian statesman who unified Germany in 1871", "Otto von Bismarck", ["Prussian", "German"], "person_history"),
    ("The empress who ruled Russia from 1762 to 1796 and expanded its borders", "Catherine the Great", ["Russian", "German"], "person_history"),
    ("The tsar who founded Saint Petersburg and modernised Russia", "Peter the Great", ["Russian"], "person_history"),
    ("The Roman emperor and Stoic philosopher who wrote the Meditations", "Marcus Aurelius", ["Roman"], "person_history"),
    ("The Carthaginian general who crossed the Alps with elephants", "Hannibal Barca", ["Carthaginian"], "person_history"),
    ("The Norman duke who conquered England in 1066", "William the Conqueror", ["Norman", "French", "English"], "person_history"),
    ("The English general who ruled as Lord Protector after the execution of Charles I", "Oliver Cromwell", ["English", "British"], "person_history"),
    ("The American founding father who flew a kite in a thunderstorm and invented the lightning rod", "Benjamin Franklin", ["American"], "person_history"),
    ("The first United States Secretary of the Treasury, killed in a duel by Aaron Burr", "Alexander Hamilton", ["American"], "person_history"),
    ("The American seamstress whose refusal to give up her bus seat in Montgomery sparked a boycott", "Rosa Parks", ["American"], "person_history"),
    ("The American abolitionist who guided enslaved people to freedom on the Underground Railroad", "Harriet Tubman", ["American"], "person_history"),
    ("The Argentine-born revolutionary who fought alongside Castro in Cuba", "Che Guevara", ["Argentine", "Argentinian", "Cuban"], "person_history"),
    ("The founder of the Republic of Turkey who abolished the caliphate", "Kemal Ataturk", ["Turkish"], "person_history"),
    ("The emperor of Ethiopia from 1930 to 1974, revered by Rastafarians", "Haile Selassie", ["Ethiopian"], "person_history"),
    ("The first president of Ghana and a leader of pan-Africanism", "Kwame Nkrumah", ["Ghanaian"], "person_history"),
    ("The first president of Kenya after independence in 1963", "Jomo Kenyatta", ["Kenyan"], "person_history"),
    ("The first president of Tanzania, known as Mwalimu", "Julius Nyerere", ["Tanzanian"], "person_history"),
    ("The Canadian hockey player nicknamed The Great One", "Wayne Gretzky", ["Canadian"], "person_sport"),
    ("The American quarterback who won seven Super Bowls", "Tom Brady", ["American"], "person_sport"),
    ("The American gymnast with the most world championship medals in history", "Simone Biles", ["American"], "person_sport"),
    ("The American sprinter and long jumper who won four gold medals at the 1984 Olympics", "Carl Lewis", ["American"], "person_sport"),
    ("The Indian cricketer who captained the national team and is known as the King", "Virat Kohli", ["Indian"], "person_sport"),
    ("The American boxer who became the youngest heavyweight champion at twenty", "Mike Tyson", ["American"], "person_sport"),
    ("The Los Angeles Lakers point guard nicknamed Magic", "Magic Johnson", ["American"], "person_sport"),
    ("The Boston Celtics forward whose rivalry with Magic Johnson defined 1980s basketball", "Larry Bird", ["American"], "person_sport"),
    ("The Los Angeles Lakers guard nicknamed the Black Mamba who died in a 2020 helicopter crash", "Kobe Bryant", ["American"], "person_sport"),
    ("The German tennis player who won the Golden Slam in 1988", "Steffi Graf", ["German"], "person_sport"),
    ("The Swedish tennis player who won five consecutive Wimbledon titles from 1976", "Bjorn Borg", ["Swedish"], "person_sport"),
    ("The American tennis player who won fourteen Grand Slam singles titles in the 1990s and retired in 2002", "Pete Sampras", ["American"], "person_sport"),
    ("The American golfer with the most major championship wins, eighteen", "Jack Nicklaus", ["American"], "person_sport"),
    ("The German Formula One driver who won seven world championships with Benetton and Ferrari", "Michael Schumacher", ["German"], "person_sport"),
    ("The British Formula One driver who won seven world championships with McLaren and Mercedes", "Lewis Hamilton", ["British"], "person_sport"),
    ("The Brazilian Formula One driver who died at Imola in 1994", "Ayrton Senna", ["Brazilian"], "person_sport"),
    ("The French footballer who headbutted Materazzi in the 2006 World Cup final", "Zinedine Zidane", ["French"], "person_sport"),
    ("The Dutch footballer who embodied Total Football at Ajax and Barcelona", "Johan Cruyff", ["Dutch"], "person_sport"),
    ("The German footballer nicknamed Der Kaiser who won the World Cup as player and coach", "Franz Beckenbauer", ["German"], "person_sport"),
    ("The Australian cricketer with a Test batting average of 99.94", "Don Bradman", ["Australian"], "person_sport"),
    ("The Australian leg-spinner who took over 700 Test wickets", "Shane Warne", ["Australian"], "person_sport"),
    ("The American swimmer who won eight gold medals at the 2008 Olympics", "Michael Phelps", ["American"], "person_sport"),
]
FRESH_PEOPLE_MORE = [  # politics, music, film, characters: (description, name, nationality, category); the name never appears in the description
    ("The British prime minister from 1979 to 1990 known as the Iron Lady", "Margaret Thatcher", ["British", "English"], "person_history"),
    ("The Labour politician who was British prime minister from 1997 to 2007", "Tony Blair", ["British", "Scottish", "English"], "person_history"),
    ("The German chancellor from 2005 to 2021, a physicist by training", "Angela Merkel", ["German"], "person_history"),
    ("The president of Russia since 2000, apart from four years as prime minister", "Vladimir Putin", ["Russian"], "person_history"),
    ("The Canadian prime minister from 2015 whose father also held the office", "Justin Trudeau", ["Canadian"], "person_history"),
    ("The Iraqi dictator overthrown by the 2003 invasion and executed in 2006", "Saddam Hussein", ["Iraqi"], "person_history"),
    ("The Chinese leader who became general secretary of the Communist Party in 2012", "Xi Jinping", ["Chinese"], "person_history"),
    ("The Vietnamese revolutionary leader after whom Saigon was renamed", "Ho Chi Minh", ["Vietnamese"], "person_history"),
    ("The Khmer Rouge leader responsible for the Cambodian genocide", "Pol Pot", ["Cambodian"], "person_history"),
    ("The Venetian merchant whose account of his travels to the court of Kublai Khan became famous", "Marco Polo", ["Venetian", "Italian"], "person_history"),
    ("The queen of Scotland executed by her cousin Elizabeth I in 1587", "Mary Queen of Scots", ["Scottish"], "person_history"),
    ("The Roman general who crossed the Rubicon and was assassinated on the Ides of March", "Julius Caesar", ["Roman"], "person_history"),
    ("The Roman general who allied with Cleopatra and lost the Battle of Actium", "Mark Antony", ["Roman"], "person_history"),
    ("The 42nd president of the United States, whose wife later ran for the office", "Bill Clinton", ["American"], "person_history"),
    ("The first woman nominated for president by a major US party, in 2016", "Hillary Clinton", ["American"], "person_history"),
    ("The first African American president of the United States", "Barack Obama", ["American"], "person_history"),
    ("The real-estate developer and television host elected US president in 2016", "Donald Trump", ["American"], "person_history"),
    ("The US president inaugurated in January 2021 at the age of seventy-eight", "Joe Biden", ["American"], "person_history"),
    ("The US vice president under George W. Bush who shot a friend while hunting", "Dick Cheney", ["American"], "person_history"),
    ("The US general who was the first African American Secretary of State", "Colin Powell", ["American"], "person_history"),
    ("The US vice president who lost the 2000 election and later made a film about climate change", "Al Gore", ["American"], "person_history"),
    ("The Vermont senator who ran for the Democratic nomination as a democratic socialist in 2016 and 2020", "Bernie Sanders", ["American"], "person_history"),
    ("The first woman to serve as Speaker of the US House of Representatives", "Nancy Pelosi", ["American"], "person_history"),
    ("The lead singer of Queen who died in 1991", "Freddie Mercury", ["British"], "person_arts"),
    ("The English singer who created the persona Ziggy Stardust", "David Bowie", ["English", "British"], "person_arts"),
    ("The American singer-songwriter who won the Nobel Prize in Literature in 2016", "Bob Dylan", ["American"], "person_arts"),
    ("The singer of Thriller, known as the King of Pop", "Michael Jackson", ["American"], "person_arts"),
    ("The American singer whose Eras Tour became the highest-grossing tour in history", "Taylor Swift", ["American"], "person_arts"),
    ("The singer who wore a dress made of raw meat to the 2010 MTV awards", "Lady Gaga", ["American"], "person_arts"),
    ("The American crooner nicknamed Ol' Blue Eyes who sang My Way", "Frank Sinatra", ["American"], "person_arts"),
    ("The New Orleans jazz trumpeter nicknamed Satchmo who sang What a Wonderful World", "Louis Armstrong", ["American"], "person_arts"),
    ("The jazz singer known as the First Lady of Song", "Ella Fitzgerald", ["American"], "person_arts"),
    ("The blind American soul musician who sang Georgia on My Mind", "Ray Charles", ["American"], "person_arts"),
    ("The country singer known as the Man in Black who recorded at Folsom Prison", "Johnny Cash", ["American"], "person_arts"),
    ("The lead singer of The Doors who died in Paris in 1971", "Jim Morrison", ["American"], "person_arts"),
    ("The Beatle who wrote Imagine and was shot in New York in 1980", "John Lennon", ["English", "British"], "person_arts"),
    ("The Beatle who wrote Yesterday and later formed Wings", "Paul McCartney", ["English", "British"], "person_arts"),
    ("The Beatle who wrote Here Comes the Sun and organised the Concert for Bangladesh", "George Harrison", ["English", "British"], "person_arts"),
    ("The guitarist of the Rolling Stones who wrote the riff of Satisfaction", "Keith Richards", ["English", "British"], "person_arts"),
    ("The singer of I Will Always Love You who died in 2012", "Whitney Houston", ["American"], "person_arts"),
    ("The Canadian pop singer discovered on YouTube who sang Baby", "Justin Bieber", ["Canadian"], "person_arts"),
    ("The rapper and producer who married Kim Kardashian and later renamed himself Ye", "Kanye West", ["American"], "person_arts"),
    ("The blonde Hollywood actress of Some Like It Hot who died in 1962", "Marilyn Monroe", ["American"], "person_arts"),
    ("The actor of Rebel Without a Cause who died in a car crash at twenty-four", "James Dean", ["American"], "person_arts"),
    ("The actor who played the Joker in the 1989 Batman film and starred in The Shining", "Jack Nicholson", ["American"], "person_arts"),
    ("The actor who plays Ethan Hunt in the Mission: Impossible films", "Tom Cruise", ["American"], "person_arts"),
    ("The actor of Fight Club who was married to Angelina Jolie", "Brad Pitt", ["American"], "person_arts"),
    ("The actor who played Han Solo and Indiana Jones", "Harrison Ford", ["American"], "person_arts"),
    ("The actor who narrated March of the Penguins and played God in Bruce Almighty", "Morgan Freeman", ["American"], "person_arts"),
    ("The actor who slapped Chris Rock at the 2022 Academy Awards", "Will Smith", ["American"], "person_arts"),
    ("The actress of Pretty Woman and Erin Brockovich", "Julia Roberts", ["American"], "person_arts"),
    ("The actress who played Katniss Everdeen in The Hunger Games", "Jennifer Lawrence", ["American"], "person_arts"),
    ("The actress who played Hermione Granger in the Harry Potter films", "Emma Watson", ["English", "British"], "person_arts"),
    ("The director of Jaws, E.T. and Schindler's List", "Steven Spielberg", ["American"], "person_arts"),
    ("The director of Inception, Interstellar and Oppenheimer", "Christopher Nolan", ["British", "American", "British-American"], "person_arts"),
    ("The director of Titanic and Avatar", "James Cameron", ["Canadian"], "person_arts"),
    ("The creator of Star Wars", "George Lucas", ["American"], "person_arts"),
    ("The New Zealand director of The Lord of the Rings trilogy", "Peter Jackson", ["New Zealander", "New Zealand"], "person_arts"),
    ("The animator who created Mickey Mouse and founded a film studio and theme parks", "Walt Disney", ["American"], "person_arts"),
    ("The director of Annie Hall and Manhattan", "Woody Allen", ["American"], "person_arts"),
    ("The director of Alien, Blade Runner and Gladiator", "Ridley Scott", ["English", "British"], "person_arts"),
]
FRESH_CHARACTERS = [  # (prompt, character) -> category character
    ("Fact: The boy wizard with a lightning-shaped scar created by J. K. Rowling is", "Harry Potter"),
    ("Fact: The British secret agent with the code number 007 is", "James Bond"),
    ("Fact: The whip-carrying archaeologist played by Harrison Ford is", "Indiana Jones"),
    ("Fact: The farm boy from Tatooine who becomes a Jedi in the original Star Wars trilogy is", "Luke Skywalker"),
    ("Fact: The black-armoured Sith lord who is Luke's father in Star Wars is", "Darth Vader"),
    ("Fact: The billionaire whose secret identity is Batman is", "Bruce Wayne"),
    ("Fact: The teenager bitten by a radioactive spider who becomes Spider-Man is", "Peter Parker"),
    ("Fact: The reporter at the Daily Planet who is secretly Superman is", "Clark Kent"),
    ("Fact: The genius billionaire whose armoured suit makes him Iron Man is", "Tony Stark"),
    ("Fact: The doughnut-loving father in the animated series set in Springfield is", "Homer Simpson"),
    ("Fact: The cartoon mouse who is the mascot of the Walt Disney Company is", "Mickey Mouse"),
    ("Fact: The short-tempered Disney duck in a sailor suit is", "Donald Duck"),
    ("Fact: The carrot-chewing Looney Tunes rabbit who says What's up, Doc is", "Bugs Bunny"),
    ("Fact: The boy who never grows up and lives in Neverland is", "Peter Pan"),
    ("Fact: The English outlaw of Sherwood Forest who robbed the rich to give to the poor is", "Robin Hood"),
    ("Fact: The legendary British king of Camelot who drew the sword from the stone is", "King Arthur"),
    ("Fact: The star-spangled shield-carrying Marvel super-soldier is", "Captain America"),
    ("Fact: The Amazon warrior princess of DC Comics with a lasso of truth is", "Wonder Woman"),
    ("Fact: The Marvel spy and Avenger played by Scarlett Johansson is", "Black Widow"),
    ("Fact: The DC hero whose power ring is charged by willpower is", "Green Lantern"),
    ("Fact: The round-headed boy in the Peanuts comic strip who owns Snoopy is", "Charlie Brown"),
    ("Fact: The time-travelling alien with a police-box spaceship in the long-running BBC series is", "Doctor Who"),
    ("Fact: The teenage girl detective in the mystery novels published since 1930 is", "Nancy Drew"),
    ("Fact: The Dickens orphan who asked for more gruel is", "Oliver Twist"),
    ("Fact: The mischievous boy in Mark Twain's novel who tricks friends into whitewashing a fence is", "Tom Sawyer"),
    ("Fact: The Shakespeare tragedy about two young lovers from feuding families in Verona is", "Romeo and Juliet"),
    ("Fact: The Shakespeare tragedy about an old king who divides his kingdom among his daughters is", "King Lear"),
    ("Fact: The jolly bearded figure who delivers presents on Christmas Eve is", "Santa Claus"),
    ("Fact: The imaginary figure who leaves money under a child's pillow in exchange for a lost tooth is the", "Tooth Fairy"),
    ("Fact: The personification of the United States as a tall man in a star-spangled top hat is", "Uncle Sam"),
    ("Fact: The nickname of New York City is the", "Big Apple"),
]
FRESH_PLACES = [  # (prompt, place)
    ("Fact: The city in Texas that is home to the Alamo is", "San Antonio"),
    ("Fact: The city in western Texas that lies across the Rio Grande from Ciudad Juarez is", "El Paso"),
    ("Fact: The capital of Louisiana is", "Baton Rouge"),
    ("Fact: The capital of Arkansas is", "Little Rock"),
    ("Fact: The capital of Iowa is", "Des Moines"),
    ("Fact: The city in California that is the home of Silicon Valley and the tenth-largest US city is", "San Jose"),
    ("Fact: The California city where Stanford University is located is", "Palo Alto"),
    ("Fact: The California city where Google has its headquarters is", "Mountain View"),
    ("Fact: The California beach city with a famous pier at the end of Route 66 is", "Santa Monica"),
    ("Fact: The California city known as the American Riviera, north of Los Angeles, is", "Santa Barbara"),
    ("Fact: The California desert resort city famous for mid-century modern architecture is", "Palm Springs"),
    ("Fact: The port city south of Los Angeles where the Queen Mary is moored is", "Long Beach"),
    ("Fact: The Florida city known as the Venice of America for its canals, north of Miami, is", "Fort Lauderdale"),
    ("Fact: The most populous city in Virginia, on the Atlantic coast, is", "Virginia Beach"),
    ("Fact: The Colorado city at the foot of Pikes Peak that hosts the US Air Force Academy is", "Colorado Springs"),
    ("Fact: The Wisconsin city whose NFL team is the Packers is", "Green Bay"),
    ("Fact: The largest city in South Dakota is", "Sioux Falls"),
    ("Fact: The Connecticut city where Yale University is located is", "New Haven"),
    ("Fact: The island east of Manhattan that contains Brooklyn and Queens is", "Long Island"),
    ("Fact: The New York City borough reached from Manhattan by a free ferry is", "Staten Island"),
    ("Fact: The smallest US state by area is", "Rhode Island"),
    ("Fact: The hook-shaped peninsula of Massachusetts popular with summer visitors is", "Cape Cod"),
    ("Fact: The southernmost city of the continental United States, at the end of the Florida Keys, is", "Key West"),
]
FRESH_COMPOUNDS_MORE = [
    ("Fact: The room of a house where the family sits and watches television is the", "living room"),
    ("Fact: A large artificial basin of water for swimming is a", "swimming pool"),
    ("Fact: The paved area where cars are left outside a shop is the", "parking lot"),
    ("Fact: The red, amber and green signal that controls cars at a junction is a", "traffic light"),
    ("Fact: The uniformed public servant who enforces the law and makes arrests is a", "police officer"),
    ("Fact: The place where you buy stamps and mail parcels is the", "post office"),
    ("Fact: The place where passengers wait to board a bus is a", "bus stop"),
    ("Fact: The place where passengers board and leave trains is a", "train station"),
    ("Fact: The place where drivers fill their cars with petrol is a", "gas station"),
    ("Fact: A large indoor complex of shops under one roof is a", "shopping mall"),
    ("Fact: The household appliance that cleans clothes is the", "washing machine"),
    ("Fact: The household appliance that sucks up dust from carpets is the", "vacuum cleaner"),
    ("Fact: The bedside device that rings to wake you up is an", "alarm clock"),
    ("Fact: The low table in front of a sofa is a", "coffee table"),
    ("Fact: The room of a house where meals are eaten is the", "dining room"),
    ("Fact: The drink squeezed from oranges is", "orange juice"),
    ("Fact: The cake with candles served at a birthday party is a", "birthday cake"),
    ("Fact: The band worn on the finger to show that one is married is a", "wedding ring"),
    ("Fact: The ring given when a couple agree to marry is an", "engagement ring"),
    ("Fact: The handheld appliance that blows hot air to dry hair is a", "hair dryer"),
    ("Fact: The ceiling device that beeps when it senses smoke is a", "smoke detector"),
    ("Fact: The strap that holds a passenger in a car seat during a crash is the", "seat belt"),
    ("Fact: The wheel a driver turns to steer a car is the", "steering wheel"),
    ("Fact: The extra wheel kept in the boot in case of a puncture is the", "spare tire"),
    ("Fact: The metal plate with letters and numbers that identifies a car is the", "license plate"),
    ("Fact: The maximum legal speed on a road is the", "speed limit"),
    ("Fact: A long line of vehicles that cannot move on a road is a", "traffic jam"),
    ("Fact: The time of day when roads are busiest with commuters is the", "rush hour"),
    ("Fact: The inflatable vest that keeps a person afloat in water is a", "life jacket"),
    ("Fact: Emergency help given to an injured person before a doctor arrives is", "first aid"),
    ("Fact: The sudden blockage of blood flow to the heart muscle is a", "heart attack"),
    ("Fact: The mild viral infection of the nose and throat that everyone gets in winter is the", "common cold"),
    ("Fact: The pain in the throat that makes swallowing hurt is a", "sore throat"),
    ("Fact: The region of space where gravity is so strong that not even light escapes is a", "black hole"),
    ("Fact: The Sun and the eight planets that orbit it form the", "solar system"),
    ("Fact: The phase of the Moon when its whole face is lit is the", "full moon"),
    ("Fact: The event in which the Moon passes between the Sun and the Earth and blocks the Sun is a", "solar eclipse"),
    ("Fact: The event in which the Earth's shadow falls on the Moon is a", "lunar eclipse"),
    ("Fact: The northernmost point on Earth is the", "North Pole"),
    ("Fact: The southernmost point on Earth is the", "South Pole"),
    ("Fact: A region of the Earth that keeps the same standard time is a", "time zone"),
    ("Fact: A year with 366 days is a", "leap year"),
    ("Fact: The average height of the ocean surface, from which altitudes are measured, is", "sea level"),
    ("Fact: A natural spring of geothermally heated water is a", "hot spring"),
    ("Fact: The ridge of coral built up by tiny marine animals in warm shallow seas is a", "coral reef"),
    ("Fact: A dense tropical forest with very heavy rainfall is a", "rain forest"),
    ("Fact: A line of connected mountains is a", "mountain range"),
    ("Fact: The head of government in the United Kingdom is the", "prime minister"),
    ("Fact: The officer who succeeds the US president if the president dies is the", "vice president"),
    ("Fact: The highest court of the United States is the", "Supreme Court"),
    ("Fact: The place where shares of public companies are bought and sold is the", "stock market"),
    ("Fact: The percentage a bank charges on a loan is the", "interest rate"),
    ("Fact: The tax that governments levy on what people earn is", "income tax"),
    ("Fact: The lowest hourly pay an employer may legally give is the", "minimum wage"),
    ("Fact: Property consisting of land and buildings is", "real estate"),
    ("Fact: The US government programme that pays pensions to retired workers is", "Social Security"),
    ("Fact: The insurance that pays for medical treatment is", "health insurance"),
    ("Fact: The rate at which one currency can be swapped for another is the", "exchange rate"),
    ("Fact: An organisation of workers that bargains with employers is a", "trade union"),
    ("Fact: The 1960s American movement led by Martin Luther King fought for", "civil rights"),
    ("Fact: The right to say what one thinks without government censorship is", "free speech"),
    ("Fact: Execution as a punishment for a crime is the", "death penalty"),
    ("Fact: A prison sentence that lasts until the prisoner dies is a", "life sentence"),
    ("Fact: The social class between the working class and the upper class is the", "middle class"),
    ("Fact: A child who has no brothers or sisters is an", "only child"),
    ("Fact: The name between a person's first name and surname is the", "middle name"),
    ("Fact: A woman's surname before she married is her", "maiden name"),
    ("Fact: A diagram showing a person's ancestors and their relationships is a", "family tree"),
    ("Fact: The classification of blood as A, B, AB or O is the", "blood type"),
    ("Fact: In baseball, a hit that lets the batter run around all the bases is a", "home run"),
    ("Fact: In basketball, the unopposed shot awarded after a foul is a", "free throw"),
    ("Fact: In football, the shot from twelve yards awarded after a foul in the box is a", "penalty kick"),
    ("Fact: Three goals by one player in a single game is a", "hat trick"),
    ("Fact: The best performance ever recorded in a sport is a", "world record"),
    ("Fact: The prize for first place at the Olympic Games is a", "gold medal"),
    ("Fact: The prize for second place at the Olympic Games is a", "silver medal"),
    ("Fact: The prize for third place at the Olympic Games is a", "bronze medal"),
    ("Fact: The sport played with small bats and a light ball on a table with a net is", "table tennis"),
    ("Fact: The sport played on skates with a puck and sticks is", "ice hockey"),
    ("Fact: The team sport played in a swimming pool with a ball and goals is", "water polo"),
    ("Fact: Karate, judo and taekwondo are examples of", "martial arts"),
    ("Fact: The sport in which skaters perform jumps and spins to music is", "figure skating"),
    ("Fact: The athletics event in which competitors jump as far as they can into a sand pit is the", "long jump"),
    ("Fact: The athletics event in which competitors jump over a horizontal bar is the", "high jump"),
    ("Fact: The athletics event in which a long flexible pole is used to clear a bar is the", "pole vault"),
    ("Fact: The athletics event in which a heavy metal ball is thrown is the", "shot put"),
    ("Fact: A bicycle with thick tyres built for rough off-road trails is a", "mountain bike"),
    ("Fact: The amusement park ride with steep drops on a track of rails is a", "roller coaster"),
    ("Fact: The red-and-white striped hook-shaped Christmas sweet is a", "candy cane"),
    ("Fact: The decorated evergreen put up in homes in December is a", "Christmas tree"),
    ("Fact: The decorated egg hidden for children to find on a spring holiday is an", "Easter egg"),
]
# (prompt, phrase) answer items, category landmark; every prompt ends where the phrase begins
FRESH_LANDMARKS = [
    ("Fact: The copper statue on Liberty Island given to the United States by France in 1886 is the", "Statue of Liberty"),
    ("Fact: The bascule bridge next to the Tower of London that opens for ships is", "Tower Bridge"),
    ("Fact: The eighteenth-century neoclassical gate that became the symbol of Berlin is the", "Brandenburg Gate"),
    ("Fact: The imperial palace complex at the centre of Beijing is the", "Forbidden City"),
    ("Fact: The highest mountain in Greece, home of the gods in mythology, is", "Mount Olympus"),
    ("Fact: The large high-altitude lake on the border of Peru and Bolivia is", "Lake Titicaca"),
    ("Fact: The salt lake between Israel and Jordan whose shore is the lowest land point on Earth is the", "Dead Sea"),
    ("Fact: The largest enclosed inland body of water on Earth, bordered by Russia and Iran, is the", "Caspian Sea"),
    ("Fact: The largest hot desert in the world, covering much of North Africa, is the", "Sahara Desert"),
    ("Fact: The large cold desert spanning northern China and southern Mongolia is the", "Gobi Desert"),
    ("Fact: The desert of southern Africa covering much of Botswana is the", "Kalahari Desert"),
    ("Fact: The driest non-polar desert on Earth, in northern Chile, is the", "Atacama Desert"),
    ("Fact: The remote Chilean island in the Pacific famous for its moai statues is", "Easter Island"),
    ("Fact: The region of the Atlantic between Florida, Bermuda and Puerto Rico where ships are said to vanish is the", "Bermuda Triangle"),
    ("Fact: The concrete arch dam on the Colorado River built during the Great Depression is the", "Hoover Dam"),
    ("Fact: The holiest shrine of Sikhism, in Amritsar, is the", "Golden Temple"),
    ("Fact: The twin skyscrapers that were the tallest buildings in the world from 1998 to 2004, in Kuala Lumpur, are the", "Petronas Towers"),
    ("Fact: The giant statue of Jesus overlooking Rio de Janeiro is", "Christ the Redeemer"),
    ("Fact: The Gothic cathedral on an island in the Seine that burned in 2019 is", "Notre Dame"),
    ("Fact: The unfinished basilica in Barcelona designed by Gaudi is the", "Sagrada Familia"),
    ("Fact: The Baroque fountain in Rome into which visitors throw coins is the", "Trevi Fountain"),
    ("Fact: The flat-topped mountain overlooking Cape Town is", "Table Mountain"),
    ("Fact: The volcano whose eruption in AD 79 buried Pompeii is", "Mount Vesuvius"),
    ("Fact: The active volcano on the east coast of Sicily is", "Mount Etna"),
    ("Fact: The mountain in eastern Turkey where Noah's Ark is said to have landed is", "Mount Ararat"),
    ("Fact: The highest mountain in Europe, in the Caucasus, is", "Mount Elbrus"),
    ("Fact: The second-deepest lake in the world, shared by Tanzania, Congo, Burundi and Zambia, is", "Lake Tanganyika"),
    ("Fact: The largest of the Great Lakes of North America by area is", "Lake Superior"),
    ("Fact: The lake on the border of Switzerland and France that Lausanne overlooks is", "Lake Geneva"),
    ("Fact: The Scottish lake famous for a legendary monster is", "Loch Ness"),
    ("Fact: The large bay of the Indian Ocean east of India and Sri Lanka is the", "Bay of Bengal"),
    ("Fact: The body of water bounded by the southern United States, Mexico and Cuba is the", "Gulf of Mexico"),
    ("Fact: The narrow strait that separates Spain from Morocco is the", "Strait of Gibraltar"),
    ("Fact: The southernmost headland of South America, feared by sailors, is", "Cape Horn"),
    ("Fact: The rocky headland near Cape Town that Portuguese sailors rounded on the way to India is the", "Cape of Good Hope"),
    ("Fact: The ancient stone circle on Salisbury Plain in England is", "Stonehenge"),
    ("Fact: The temple complex in Cambodia that is the largest religious monument in the world is", "Angkor Wat"),
    ("Fact: The Roman amphitheatre in the centre of Rome is the", "Colosseum"),
    ("Fact: The Moorish palace and fortress in Granada is the", "Alhambra"),
    ("Fact: The large sandstone monolith in central Australia sacred to the Anangu is", "Uluru"),
    ("Fact: The waterfall on the Zambezi River on the border of Zambia and Zimbabwe is", "Victoria Falls"),
    ("Fact: The steep-sided gorge carved by the Colorado River in Arizona is the", "Grand Canyon"),
    ("Fact: The bridge across the strait at the entrance to San Francisco Bay, painted orange, is the", "Golden Gate Bridge"),
    ("Fact: The tallest building in the world, in Dubai, is the", "Burj Khalifa"),
    ("Fact: The mausoleum of white marble in Agra built by Shah Jahan is the", "Taj Mahal"),
    ("Fact: The lost Inca city high in the Peruvian Andes is", "Machu Picchu"),
    ("Fact: The performing arts centre with white sail-shaped shells on Sydney Harbour is the", "Sydney Opera House"),
    ("Fact: The world's largest coral reef system, off the coast of Queensland, is the", "Great Barrier Reef"),
    ("Fact: The long fortification built across northern China to keep out nomadic invaders is the", "Great Wall of China"),
    ("Fact: The waterfalls on the border of Ontario and New York State are", "Niagara Falls"),
    ("Fact: The granite carving of four US presidents in South Dakota is", "Mount Rushmore"),
    ("Fact: The tallest mountain in Japan, a sacred snow-capped volcano, is", "Mount Fuji"),
]
FRESH_EVENTS = [
    ("Fact: The 1853-1856 war in which Britain, France and the Ottoman Empire fought Russia was the", "Crimean War"),
    ("Fact: The 1899-1902 war between the British Empire and the Afrikaner republics of South Africa was the", "Boer War"),
    ("Fact: The 1918 influenza pandemic that killed tens of millions is known as the", "Spanish Flu"),
    ("Fact: The Japanese attack on 7 December 1941 that brought the United States into the Second World War was on", "Pearl Harbor"),
    ("Fact: The secret American programme that built the first atomic bombs was the", "Manhattan Project"),
    ("Fact: The American programme of aid for the reconstruction of Western Europe after 1948 was the", "Marshall Plan"),
    ("Fact: The barrier that divided East and West Berlin from 1961 to 1989 was the", "Berlin Wall"),
    ("Fact: Churchill's name for the boundary dividing Soviet-controlled Europe from the West was the", "Iron Curtain"),
    ("Fact: The failed 1961 CIA-backed invasion of Cuba is known as the", "Bay of Pigs"),
    ("Fact: The first ten amendments to the United States Constitution are known as the", "Bill of Rights"),
    ("Fact: The 1776 document in which the thirteen American colonies broke with Britain is the", "Declaration of Independence"),
    ("Fact: The 1919 peace treaty that formally ended the First World War with Germany was the", "Treaty of Versailles"),
    ("Fact: The 1945 meeting of Truman, Stalin and Churchill near Berlin was the", "Potsdam Conference"),
    ("Fact: The 1814-1815 meeting of European powers that redrew the map after Napoleon was the", "Congress of Vienna"),
    ("Fact: The 1648 treaties that ended the Thirty Years' War are known as the", "Peace of Westphalia"),
    ("Fact: The 1803 purchase by which the United States bought a vast territory from France was the", "Louisiana Purchase"),
    ("Fact: The forced removal of the Cherokee and other nations to the west in the 1830s is known as the", "Trail of Tears"),
    ("Fact: Mao's 1958-1962 campaign of rapid industrialisation that led to famine was the", "Great Leap Forward"),
    ("Fact: The 1966-1976 upheaval launched by Mao Zedong in China was the", "Cultural Revolution"),
    ("Fact: The 1868 restoration of imperial rule in Japan that began its modernisation was the", "Meiji Restoration"),
    ("Fact: The 1899-1901 anti-foreign uprising in China put down by an eight-nation alliance was the", "Boxer Rebellion"),
    ("Fact: Gandhi's 1930 march to the sea to protest the British monopoly was the", "Salt March"),
    ("Fact: The 1916 armed insurrection in Dublin against British rule was the", "Easter Rising"),
    ("Fact: The 1968 period of liberalisation in Czechoslovakia crushed by Soviet tanks was the", "Prague Spring"),
    ("Fact: The peaceful 1989 overthrow of communist rule in Czechoslovakia was the", "Velvet Revolution"),
    ("Fact: The wave of uprisings across the Middle East and North Africa that began in Tunisia in 2010 was the", "Arab Spring"),
    ("Fact: The 1973 war launched by Egypt and Syria against Israel on a Jewish holy day was the", "Yom Kippur War"),
    ("Fact: The 1956 conflict in which Britain, France and Israel attacked Egypt over a nationalised canal was the", "Suez Crisis"),
    ("Fact: The 1066 battle in which William of Normandy defeated King Harold was the", "Battle of Hastings"),
    ("Fact: The 1815 battle in Belgium at which Napoleon was finally defeated was the", "Battle of Waterloo"),
    ("Fact: The 1805 naval battle in which Nelson destroyed the French and Spanish fleets was the", "Battle of Trafalgar"),
    ("Fact: The 1863 battle in Pennsylvania that was the turning point of the American Civil War was the", "Battle of Gettysburg"),
    ("Fact: The 1942-1943 battle on the Volga that turned the tide on the Eastern Front was the", "Battle of Stalingrad"),
    ("Fact: The 1942 naval battle in the Pacific in which the US Navy sank four Japanese aircraft carriers was the", "Battle of Midway"),
    ("Fact: The 480 BC battle in which three hundred Spartans held a mountain pass against the Persians was the", "Battle of Thermopylae"),
    ("Fact: The 490 BC battle in which the Athenians defeated the first Persian invasion was the", "Battle of Marathon"),
    ("Fact: The 1836 siege in Texas in which every defender of a mission was killed by Santa Anna's army was the", "Battle of the Alamo"),
    ("Fact: The 1770 shooting of colonists by British soldiers in a Massachusetts city is known as the", "Boston Massacre"),
    ("Fact: The 1917 seizure of power in Russia by Lenin's Bolsheviks was the", "October Revolution"),
    ("Fact: The 1848 discovery of gold at Sutter's Mill set off the", "California Gold Rush"),
    ("Fact: The annual prizes established by the will of the Swedish inventor of dynamite are the", "Nobel Prize"),
    ("Fact: The quadrennial international multi-sport competition revived in Athens in 1896 is the", "Olympic Games"),
    ("Fact: The championship game of the National Football League is the", "Super Bowl"),
    ("Fact: The three-week cycling race around France held every July is the", "Tour de France"),
    ("Fact: The trophy awarded to the champion of the National Hockey League is the", "Stanley Cup"),
    ("Fact: The top European club football competition organised by UEFA is the", "Champions League"),
    ("Fact: The top division of English football is the", "Premier League"),
    ("Fact: The highest class of single-seater motor racing is", "Formula One"),
    ("Fact: The horse race run every May at Churchill Downs in Louisville is the", "Kentucky Derby"),
    ("Fact: The annual marathon held in Massachusetts every April since 1897 is the", "Boston Marathon"),
    ("Fact: The international men's team tennis competition first held in 1900 is the", "Davis Cup"),
    ("Fact: The biennial golf competition between Europe and the United States is the", "Ryder Cup"),
    ("Fact: The period of European history between the fall of Rome and the Renaissance is the", "Middle Ages"),
    ("Fact: The prehistoric period when humans made tools from stone is the", "Stone Age"),
    ("Fact: The prehistoric period named after the alloy of copper and tin used for tools is the", "Bronze Age"),
    ("Fact: The prehistoric period that followed the Bronze Age is the", "Iron Age"),
    ("Fact: The most recent geological period of extensive glaciation is the", "Ice Age"),
    ("Fact: The specialised agency of the United Nations responsible for international public health is the", "World Health Organization"),
    ("Fact: The political and economic union of 27 European countries with its headquarters in Brussels is the", "European Union"),
    ("Fact: The humanitarian organisation founded by Henry Dunant in 1863 with a symbol of a red emblem on white is the", "Red Cross"),
    ("Fact: The Wall Street stock market crash and the decade of hardship that followed it are known as the", "Great Depression"),
    ("Fact: The 1937 airship disaster at Lakehurst, New Jersey was the", "Hindenburg disaster"),
    ("Fact: The 1986 nuclear accident in Soviet Ukraine happened at", "Chernobyl"),
    ("Fact: The 1912 sinking of a British liner on its maiden voyage after hitting an iceberg was the sinking of the", "Titanic"),
]
FRESH_COMPOUNDS = [  # common two-word nouns (category compound), as PHRASE_MORE["compound"]
    ("Fact: The weak acid that gives lemons and limes their sour taste is", "citric acid"),
    ("Fact: The acid that builds up in muscles during hard exercise is", "lactic acid"),
    ("Fact: The building blocks of proteins are", "amino acids"),
    ("Fact: The chemical name of vitamin C is", "ascorbic acid"),
    ("Fact: The B vitamin that pregnant women take to prevent birth defects is", "folic acid"),
    ("Fact: Sodium bicarbonate, used to make cakes rise, is commonly called", "baking soda"),
    ("Fact: Solid carbon dioxide, used to keep things cold, is called", "dry ice"),
    ("Fact: Nitrous oxide, used by dentists, is commonly called", "laughing gas"),
    ("Fact: The fossil fuel consisting mainly of methane is", "natural gas"),
    ("Fact: Unrefined petroleum pumped from the ground is called", "crude oil"),
    ("Fact: The sugary sap of a North American tree that is boiled into a breakfast syrup is", "maple syrup"),
    ("Fact: The salty fermented condiment made from soybeans is", "soy sauce"),
    ("Fact: The spice made by grinding the dried berries of Piper nigrum is", "black pepper"),
    ("Fact: Sugar that keeps some of its molasses and is used in cookies is", "brown sugar"),
    ("Fact: A payment card that takes money directly from your bank account is a", "debit card"),
    ("Fact: The storage device in a computer with spinning magnetic platters is the", "hard drive"),
    ("Fact: The flexible magnetic storage medium in a square plastic case used before CDs was the", "floppy disk"),
    ("Fact: The optical disc format introduced in 1982 for storing music is the", "compact disc"),
    ("Fact: The device that generates electricity from controlled nuclear fission is a", "nuclear reactor"),
    ("Fact: The machine that powered the Industrial Revolution by turning boiling water into motion was the", "steam engine"),
    ("Fact: The reusable American spacecraft that flew from 1981 to 2011 was the", "space shuttle"),
    ("Fact: The habitable satellite orbiting Earth that has been continuously crewed since 2000 is the International", "Space Station"),
    ("Fact: The dense hot remnant left when a Sun-like star sheds its outer layers is a", "white dwarf"),
    ("Fact: The huge cool star that the Sun will become in about five billion years is a", "red giant"),
    ("Fact: The extremely dense collapsed core left after a supernova, made mostly of neutrons, is a", "neutron star"),
    ("Fact: The invisible mass that holds galaxies together but does not emit light is called", "dark matter"),
    ("Fact: The mysterious force driving the accelerating expansion of the universe is called", "dark energy"),
    ("Fact: The distance light travels in one year is called a", "light year"),
    ("Fact: The boundary around a black hole beyond which nothing can escape is the", "event horizon"),
    ("Fact: The theory in which fundamental particles are tiny vibrating one-dimensional objects is", "string theory"),
    ("Fact: Einstein's 1915 theory of gravity as the curvature of spacetime is", "general relativity"),
    ("Fact: The theory that the Earth's outer shell is divided into moving plates is", "plate tectonics"),
    ("Fact: Rain made acidic by sulfur and nitrogen pollution is called", "acid rain"),
    ("Fact: The continuous movement of water between the oceans, the air and the land is the", "water cycle"),
    ("Fact: The thin layer that surrounds every cell and controls what enters it is the", "cell membrane"),
    ("Fact: The rigid outer layer found in plant cells but not animal cells is the", "cell wall"),
    ("Fact: The folds of tissue in the larynx that vibrate to produce the voice are the", "vocal cords"),
    ("Fact: The nerve that carries signals from the eye to the brain is the", "optic nerve"),
    ("Fact: The long coiled part of the gut where most nutrients are absorbed is the", "small intestine"),
    ("Fact: The number of times the heart beats per minute is the", "heart rate"),
    ("Fact: The bacterial disease that causes violent coughing fits ending in a gasp, also called pertussis, is", "whooping cough"),
    ("Fact: The mosquito-borne disease that turns the skin yellow, historically deadly in Panama, is", "yellow fever"),
    ("Fact: The allergic reaction to pollen that causes sneezing in spring is", "hay fever"),
    ("Fact: The disease spread by the tsetse fly that causes drowsiness is", "sleeping sickness"),
    ("Fact: The inherited disease that clogs the lungs with thick mucus is", "cystic fibrosis"),
    ("Fact: The autoimmune disease in which the immune system attacks the myelin of nerves is", "multiple sclerosis"),
    ("Fact: The puzzle made of interlocking pieces that form a picture is a", "jigsaw puzzle"),
    ("Fact: The word game with a grid of black and white squares and numbered clues is a", "crossword puzzle"),
    ("Fact: The gambling machine with spinning reels and a lever is a", "slot machine"),
    ("Fact: The party game in which players walk around seats while music plays and sit when it stops is", "musical chairs"),
    ("Fact: The exercise in which you swing a rope over your head and hop over it is", "jump rope"),
    ("Fact: The plastic ring you spin around your waist is a", "hula hoop"),
    ("Fact: The contest in which two teams pull on opposite ends of a rope is", "tug of war"),
    ("Fact: The children's game in which one player counts while the others conceal themselves is", "hide and seek"),
    ("Fact: The frozen dessert made of sweetened cream is", "ice cream"),
    ("Fact: The sandwich of a sausage in a sliced bun is a", "hot dog"),
    ("Fact: Deep-fried strips of potato served with burgers are", "french fries"),
    ("Fact: The spun-sugar confection sold at fairs is", "cotton candy"),
    ("Fact: The spread made from ground roasted peanuts is", "peanut butter"),
    ("Fact: The brief flash of light in the sky when a meteoroid burns up is a", "shooting star"),
    ("Fact: The tide with the greatest range, occurring at new and full moon, is a", "spring tide"),
    ("Fact: The line where the sky appears to meet the ground is the", "horizon"),
    ("Fact: The colourful arc that appears when sunlight passes through rain is a", "rainbow"),
    ("Fact: The rotating column of air that touches the ground during a severe thunderstorm is a", "tornado"),
    ("Fact: The warm ocean current that flows from the Gulf of Mexico across the Atlantic is the", "Gulf Stream"),
    ("Fact: The star pattern of seven bright stars in Ursa Major is the", "Big Dipper"),
    ("Fact: The star that lies almost exactly above the Earth's north pole is the", "North Star"),
    ("Fact: The glowing bands of light in the night sky near the north pole are the", "Northern Lights"),
    ("Fact: The imaginary line around the middle of the Earth at zero degrees latitude is the", "equator"),
]
# bridges hiding a fresh landmark / event behind its description: (prompt ending " is" / " was", hidden phrase, target)
FRESH_LANDMARK_BRIDGES = [
    ("Fact: The country where the copper statue given by France in 1886 stands on an island in a harbour is the", "Statue of Liberty", ["United States", "USA", "US", "America"]),
    ("Fact: The city where the neoclassical gate that symbolised a divided country and its reunification stands is", "Brandenburg Gate", ["Berlin"]),
    ("Fact: The country where the imperial palace complex with a moat and 9,000 rooms at the centre of its capital is", "Forbidden City", ["China"]),
    ("Fact: The country in which the highest mountain of the ancient Greek gods stands is", "Mount Olympus", ["Greece"]),
    ("Fact: The continent on which the driest non-polar desert on Earth lies is", "Atacama Desert", ["South America"]),
    ("Fact: The country that owns the remote Pacific island famous for its giant stone heads is", "Easter Island", ["Chile"]),
    ("Fact: The river dammed by the concrete arch dam built during the Great Depression is the", "Hoover Dam", ["Colorado"]),
    ("Fact: The city where the holiest shrine of Sikhism, plated with gold, stands is", "Golden Temple", ["Amritsar"]),
    ("Fact: The country where the twin skyscrapers that were the world's tallest buildings from 1998 to 2004 stand is", "Petronas Towers", ["Malaysia"]),
    ("Fact: The city overlooked by a giant statue of Jesus with outstretched arms is", "Christ the Redeemer", ["Rio de Janeiro", "Rio"]),
    ("Fact: The river on whose island the Gothic cathedral that burned in April 2019 stands is the", "Notre Dame", ["Seine"]),
    ("Fact: The country where the unfinished basilica designed by Gaudi stands is", "Sagrada Familia", ["Spain"]),
    ("Fact: The country where the volcano that buried Pompeii in AD 79 stands is", "Mount Vesuvius", ["Italy"]),
    ("Fact: The island on whose east coast Europe's most active volcano stands is", "Mount Etna", ["Sicily"]),
    ("Fact: The country where the mountain on which Noah's Ark is said to have landed stands is", "Mount Ararat", ["Turkey"]),
    ("Fact: The mountain range in which the highest peak of Europe stands is the", "Mount Elbrus", ["Caucasus"]),
    ("Fact: The country where the lake famous for a legendary monster lies is", "Loch Ness", ["Scotland", "United Kingdom", "UK"]),
    ("Fact: The continent whose southernmost headland was feared by sailors rounding it is", "Cape Horn", ["South America"]),
    ("Fact: The country where the ancient stone circle on Salisbury Plain stands is", "Stonehenge", ["England", "United Kingdom", "UK"]),
    ("Fact: The country where the largest religious monument in the world, a temple complex, stands is", "Angkor Wat", ["Cambodia"]),
    ("Fact: The country where the flat-topped mountain overlooking a city at the southern tip of Africa stands is", "Table Mountain", ["South Africa"]),
    ("Fact: The country where the white marble mausoleum built by Shah Jahan stands is", "Taj Mahal", ["India"]),
    ("Fact: The country where the lost Inca city high in the Andes lies is", "Machu Picchu", ["Peru"]),
    ("Fact: The country where the tallest building in the world stands is the", "Burj Khalifa", ["United Arab Emirates", "UAE"]),
    ("Fact: The state in which the granite carving of four presidents stands is", "Mount Rushmore", ["South Dakota"]),
    ("Fact: The river that carved the steep-sided gorge in Arizona is the", "Grand Canyon", ["Colorado"]),
]
# (description, full name) -> bridge_person_*_nationality is built from FRESH_PEOPLE with the " was" template

# ------------------------------------------------------------------ track A tables (sub-word) --------------------
# countries NOT in make_h2a_items.COUNTRIES: (country, capital, currency, official language, continent); None = ambiguous
FRESH_COUNTRIES = [
    ("Fiji", "Suva", "dollar", None, "Oceania"), ("Samoa", "Apia", "tala", "Samoan", "Oceania"), ("Tonga", "Nukualofa", "paanga", "Tongan", "Oceania"),
    ("Vanuatu", "Port Vila", "vatu", None, "Oceania"), ("Kiribati", "Tarawa", "dollar", None, "Oceania"), ("Nauru", "Yaren", "dollar", "Nauruan", "Oceania"),
    ("Tuvalu", "Funafuti", "dollar", None, "Oceania"), ("Palau", "Ngerulmud", "dollar", None, "Oceania"), ("Micronesia", "Palikir", "dollar", "English", "Oceania"),
    ("Papua New Guinea", "Port Moresby", "kina", None, "Oceania"), ("Solomon Islands", "Honiara", "dollar", "English", "Oceania"), ("Brunei", "Bandar Seri Begawan", "dollar", "Malay", "Asia"),
    ("Timor-Leste", "Dili", "dollar", None, "Asia"), ("Yemen", "Sanaa", "rial", "Arabic", "Asia"), ("Kuwait", "Kuwait City", "dinar", "Arabic", "Asia"),
    ("Georgia", "Tbilisi", "lari", "Georgian", None), ("Cyprus", "Nicosia", "euro", None, None), ("Luxembourg", "Luxembourg City", "euro", None, "Europe"),
    ("Liechtenstein", "Vaduz", "franc", "German", "Europe"), ("Monaco", "Monaco", "euro", "French", "Europe"), ("Andorra", "Andorra la Vella", "euro", "Catalan", "Europe"),
    ("San Marino", "San Marino", "euro", "Italian", "Europe"), ("Kosovo", "Pristina", "euro", None, "Europe"), ("Mauritius", "Port Louis", "rupee", None, "Africa"),
    ("Seychelles", "Victoria", "rupee", None, "Africa"), ("Comoros", "Moroni", "franc", None, "Africa"), ("Djibouti", "Djibouti", "franc", None, "Africa"),
    ("Burundi", "Gitega", "franc", None, "Africa"), ("Lesotho", "Maseru", "loti", None, "Africa"), ("Eswatini", "Mbabane", "lilangeni", None, "Africa"),
    ("Gabon", "Libreville", "franc", "French", "Africa"), ("Cameroon", "Yaounde", "franc", None, "Africa"), ("Chad", "N'Djamena", "franc", None, "Africa"),
    ("Benin", "Porto-Novo", "franc", "French", "Africa"), ("Togo", "Lome", "franc", "French", "Africa"), ("Guinea", "Conakry", "franc", "French", "Africa"),
    ("Gambia", "Banjul", "dalasi", "English", "Africa"), ("Mauritania", "Nouakchott", "ouguiya", "Arabic", "Africa"), ("Cape Verde", "Praia", "escudo", "Portuguese", "Africa"),
    ("Equatorial Guinea", "Malabo", "franc", None, "Africa"), ("Central African Republic", "Bangui", "franc", None, "Africa"), ("Republic of the Congo", "Brazzaville", "franc", "French", "Africa"),
    ("Guyana", "Georgetown", "dollar", "English", "South America"), ("Suriname", "Paramaribo", "dollar", "Dutch", "South America"), ("Belize", "Belmopan", "dollar", "English", None),
    ("Honduras", "Tegucigalpa", "lempira", "Spanish", None), ("Nicaragua", "Managua", "cordoba", "Spanish", None), ("Panama", "Panama City", "balboa", "Spanish", None),
    ("Haiti", "Port-au-Prince", "gourde", None, None), ("Jamaica", "Kingston", "dollar", "English", None), ("Barbados", "Bridgetown", "dollar", "English", None),
    ("Bahamas", "Nassau", "dollar", "English", None), ("Grenada", "St. George's", "dollar", "English", None), ("Dominica", "Roseau", "dollar", "English", None),
    ("Saint Lucia", "Castries", "dollar", "English", None), ("Antigua and Barbuda", "St. John's", "dollar", "English", None),
]
# elements NOT in make_h2a_items.ELEMENTS (checked by the filter): (symbol, name, atomic number)
FRESH_ELEMENTS = [("Li", "lithium", 3), ("Be", "beryllium", 4), ("B", "boron", 5), ("F", "fluorine", 9), ("Ne", "neon", 10), ("Mg", "magnesium", 12), ("Al", "aluminium", 13),
                  ("Si", "silicon", 14), ("P", "phosphorus", 15), ("S", "sulfur", 16), ("Cl", "chlorine", 17), ("Ar", "argon", 18), ("K", "potassium", 19), ("Ca", "calcium", 20),
                  ("Sc", "scandium", 21), ("Ti", "titanium", 22), ("V", "vanadium", 23), ("Cr", "chromium", 24), ("Mn", "manganese", 25), ("Fe", "iron", 26), ("Co", "cobalt", 27),
                  ("Ni", "nickel", 28), ("Cu", "copper", 29), ("Zn", "zinc", 30), ("Ga", "gallium", 31), ("Ge", "germanium", 32), ("As", "arsenic", 33), ("Se", "selenium", 34),
                  ("Br", "bromine", 35), ("Kr", "krypton", 36), ("Rb", "rubidium", 37), ("Sr", "strontium", 38), ("Y", "yttrium", 39), ("Zr", "zirconium", 40), ("Nb", "niobium", 41),
                  ("Mo", "molybdenum", 42), ("Tc", "technetium", 43), ("Ru", "ruthenium", 44), ("Rh", "rhodium", 45), ("Pd", "palladium", 46), ("Ag", "silver", 47), ("Cd", "cadmium", 48),
                  ("In", "indium", 49), ("Sn", "tin", 50), ("Sb", "antimony", 51), ("Te", "tellurium", 52), ("I", "iodine", 53), ("Xe", "xenon", 54), ("Cs", "caesium", 55), ("Ba", "barium", 56),
                  ("La", "lanthanum", 57), ("Ce", "cerium", 58), ("Pr", "praseodymium", 59), ("Nd", "neodymium", 60), ("Pm", "promethium", 61), ("Sm", "samarium", 62), ("Eu", "europium", 63),
                  ("Gd", "gadolinium", 64), ("Tb", "terbium", 65), ("Dy", "dysprosium", 66), ("Ho", "holmium", 67), ("Er", "erbium", 68), ("Tm", "thulium", 69), ("Yb", "ytterbium", 70),
                  ("Lu", "lutetium", 71), ("Hf", "hafnium", 72), ("Ta", "tantalum", 73), ("W", "tungsten", 74), ("Re", "rhenium", 75), ("Os", "osmium", 76), ("Ir", "iridium", 77),
                  ("Pt", "platinum", 78), ("Au", "gold", 79), ("Hg", "mercury", 80), ("Tl", "thallium", 81), ("Pb", "lead", 82), ("Bi", "bismuth", 83), ("Po", "polonium", 84),
                  ("At", "astatine", 85), ("Rn", "radon", 86), ("Fr", "francium", 87), ("Ra", "radium", 88), ("Ac", "actinium", 89), ("Th", "thorium", 90), ("Pa", "protactinium", 91),
                  ("U", "uranium", 92), ("Np", "neptunium", 93), ("Pu", "plutonium", 94), ("Am", "americium", 95), ("Cm", "curium", 96), ("Bk", "berkelium", 97), ("Cf", "californium", 98),
                  ("Es", "einsteinium", 99), ("Fm", "fermium", 100), ("Md", "mendelevium", 101), ("No", "nobelium", 102), ("Lr", "lawrencium", 103), ("Rf", "rutherfordium", 104),
                  ("Db", "dubnium", 105), ("Sg", "seaborgium", 106), ("Bh", "bohrium", 107), ("Hs", "hassium", 108), ("Mt", "meitnerium", 109), ("Ds", "darmstadtium", 110),
                  ("Rg", "roentgenium", 111), ("Cn", "copernicium", 112), ("Nh", "nihonium", 113), ("Fl", "flerovium", 114), ("Mc", "moscovium", 115), ("Lv", "livermorium", 116),
                  ("Ts", "tennessine", 117), ("Og", "oganesson", 118)]
FRESH_ANIMALS = [  # (prompt after "Fact: ", animal)
    ("The largest rodent in the world, native to South America, is the", "capybara"), ("The marsupial that carries its young in a pouch and hops across Australia is the", "kangaroo"),
    ("The black-and-white flightless bird of the Antarctic is the", "penguin"), ("The spotted big cat of Africa known for its speed, second only to the cheetah among cats, that hides prey in trees is the", "leopard"),
    ("The striped horse-like animal of the African savanna is the", "zebra"), ("The Arctic bear with white fur is the", "polar bear"),
    ("The largest living lizard, found on Indonesian islands, is the", "Komodo dragon"), ("The slow-moving South American mammal that hangs upside down from trees is the", "sloth"),
    ("The armoured mammal of the Americas that can roll into a ball is the", "armadillo"), ("The Australian mammal with a duck's bill that lays eggs is the", "platypus"),
    ("The venomous snake of India with a hood, charmed by flute players, is the", "cobra"), ("The largest snake by weight, found in the Amazon, is the", "anaconda"),
    ("The bird of prey that is the national bird of the United States is the", "bald eagle"), ("The colourful tropical bird that can mimic human speech is the", "parrot"),
    ("The African bird that is the largest and fastest-running bird in the world is the", "ostrich"), ("The long-legged pink wading bird that stands on one leg is the", "flamingo"),
    ("The nocturnal bird of prey that can turn its head almost all the way round is the", "owl"), ("The tiny bird that hovers while drinking nectar is the", "hummingbird"),
    ("The web-footed aquatic mammal that builds dams is the", "beaver"), ("The spiny mammal that rolls into a ball when threatened is the", "hedgehog"),
    ("The black-and-white striped animal that sprays a foul-smelling liquid is the", "skunk"), ("The masked nocturnal North American mammal that washes its food is the", "raccoon"),
    ("The large-antlered deer of northern forests, the largest of the deer family, is the", "moose"), ("The humped desert animal that stores fat for long journeys is the", "camel"),
    ("The African animal with a very long neck is the", "giraffe"), ("The horned African animal poached for its horn is the", "rhinoceros"),
    ("The river-dwelling African animal whose name means river horse is the", "hippopotamus"), ("The largest land animal, with a trunk and tusks, is the", "elephant"),
    ("The intelligent marine mammal that communicates with clicks and whistles is the", "dolphin"), ("The largest animal that has ever lived is the", "blue whale"),
    ("The eight-armed sea creature that squirts ink is the", "octopus"), ("The sea creature with a hard shell and large front claws that turns red when cooked is the", "lobster"),
    ("The bear of China that eats bamboo is the", "giant panda"), ("The largest primate, native to the forests of central Africa, is the", "gorilla"),
    ("The red-haired great ape of Borneo and Sumatra is the", "orangutan"), ("The Madagascan primate with a long striped tail is the", "lemur"),
    ("The tailless amphibian that starts life as a tadpole and croaks is the", "frog"), ("The reptile with a hard shell that lives for over a hundred years is the", "tortoise"),
    ("The reptile that changes colour and catches insects with a long tongue is the", "chameleon"), ("The insect that produces honey is the", "honeybee"),
    ("The insect that begins as a caterpillar and emerges from a chrysalis is the", "butterfly"), ("The eight-legged arachnid that spins a web is the", "spider"),
    ("The mammal that flies and sleeps hanging upside down is the", "bat"), ("The largest bird of prey in the Andes is the", "condor"),
    ("The Australian bird that laughs is the", "kookaburra"), ("The large flightless bird of Australia, second only to the ostrich in size, is the", "emu"),
]
FRESH_WORDS = [  # (English, es, fr, de, it, pt) - words NOT in make_h2a_items.TRANSLATIONS (checked by the filter)
    ("bridge", "puente", "pont", "Brücke", "ponte", "ponte"), ("window", "ventana", "fenêtre", "Fenster", "finestra", "janela"),
    ("kitchen", "cocina", "cuisine", "Küche", "cucina", "cozinha"), ("island", "isla", "île", "Insel", "isola", "ilha"),
    ("forest", "bosque", "forêt", "Wald", "foresta", "floresta"), ("mountain", "montaña", "montagne", "Berg", "montagna", "montanha"),
    ("river", "río", "rivière", "Fluss", "fiume", "rio"), ("cheese", "queso", "fromage", "Käse", "formaggio", "queijo"),
    ("bread", "pan", "pain", "Brot", "pane", "pão"), ("butter", "mantequilla", "beurre", "Butter", "burro", "manteiga"),
    ("chicken", "pollo", "poulet", "Huhn", "pollo", "frango"), ("grape", "uva", "raisin", "Traube", "uva", "uva"),
    ("strawberry", "fresa", "fraise", "Erdbeere", "fragola", "morango"), ("onion", "cebolla", "oignon", "Zwiebel", "cipolla", "cebola"),
    ("carrot", "zanahoria", "carotte", "Karotte", "carota", "cenoura"), ("mushroom", "champiñón", "champignon", "Pilz", "fungo", "cogumelo"),
    ("teacher", "maestro", "professeur", "Lehrer", "insegnante", "professor"), ("doctor", "médico", "médecin", "Arzt", "medico", "médico"),
    ("lawyer", "abogado", "avocat", "Anwalt", "avvocato", "advogado"), ("soldier", "soldado", "soldat", "Soldat", "soldato", "soldado"),
    ("brother", "hermano", "frère", "Bruder", "fratello", "irmão"), ("sister", "hermana", "sœur", "Schwester", "sorella", "irmã"),
    ("grandfather", "abuelo", "grand-père", "Großvater", "nonno", "avô"), ("husband", "marido", "mari", "Ehemann", "marito", "marido"),
    ("wednesday", "miércoles", "mercredi", "Mittwoch", "mercoledì", "quarta-feira"), ("thursday", "jueves", "jeudi", "Donnerstag", "giovedì", "quinta-feira"),
    ("saturday", "sábado", "samedi", "Samstag", "sabato", "sábado"), ("autumn", "otoño", "automne", "Herbst", "autunno", "outono"),
    ("spring", "primavera", "printemps", "Frühling", "primavera", "primavera"), ("thunder", "trueno", "tonnerre", "Donner", "tuono", "trovão"),
    ("lightning", "relámpago", "éclair", "Blitz", "fulmine", "relâmpago"), ("cloud", "nube", "nuage", "Wolke", "nuvola", "nuvem"),
    ("snow", "nieve", "neige", "Schnee", "neve", "neve"), ("umbrella", "paraguas", "parapluie", "Regenschirm", "ombrello", "guarda-chuva"),
    ("shoe", "zapato", "chaussure", "Schuh", "scarpa", "sapato"), ("glove", "guante", "gant", "Handschuh", "guanto", "luva"),
    ("pillow", "almohada", "oreiller", "Kissen", "cuscino", "travesseiro"), ("blanket", "manta", "couverture", "Decke", "coperta", "cobertor"),
    ("candle", "vela", "bougie", "Kerze", "candela", "vela"), ("mirror", "espejo", "miroir", "Spiegel", "specchio", "espelho"),
    ("scissors", "tijeras", "ciseaux", "Schere", "forbici", "tesoura"), ("hammer", "martillo", "marteau", "Hammer", "martello", "martelo"),
    ("wheel", "rueda", "roue", "Rad", "ruota", "roda"), ("bicycle", "bicicleta", "vélo", "Fahrrad", "bicicletta", "bicicleta"),
    ("airplane", "avión", "avion", "Flugzeug", "aereo", "avião"), ("ship", "barco", "navire", "Schiff", "nave", "navio"),
    ("castle", "castillo", "château", "Schloss", "castello", "castelo"), ("church", "iglesia", "église", "Kirche", "chiesa", "igreja"),
    ("hospital", "hospital", "hôpital", "Krankenhaus", "ospedale", "hospital"), ("library", "biblioteca", "bibliothèque", "Bibliothek", "biblioteca", "biblioteca"),
    ("newspaper", "periódico", "journal", "Zeitung", "giornale", "jornal"), ("letter", "carta", "lettre", "Brief", "lettera", "carta"),
    ("silver", "plata", "argent", "Silber", "argento", "prata"), ("iron", "hierro", "fer", "Eisen", "ferro", "ferro"),
    ("wood", "madera", "bois", "Holz", "legno", "madeira"), ("glass", "vidrio", "verre", "Glas", "vetro", "vidro"),
    ("salt", "sal", "sel", "Salz", "sale", "sal"), ("sugar", "azúcar", "sucre", "Zucker", "zucchero", "açúcar"),
    ("honey", "miel", "miel", "Honig", "miele", "mel"), ("pepper", "pimienta", "poivre", "Pfeffer", "pepe", "pimenta"),
    ("stomach", "estómago", "estomac", "Magen", "stomaco", "estômago"), ("shoulder", "hombro", "épaule", "Schulter", "spalla", "ombro"),
    ("finger", "dedo", "doigt", "Finger", "dito", "dedo"), ("tooth", "diente", "dent", "Zahn", "dente", "dente"),
    ("tongue", "lengua", "langue", "Zunge", "lingua", "língua"), ("knee", "rodilla", "genou", "Knie", "ginocchio", "joelho"),
    ("dream", "sueño", "rêve", "Traum", "sogno", "sonho"), ("truth", "verdad", "vérité", "Wahrheit", "verità", "verdade"),
    ("freedom", "libertad", "liberté", "Freiheit", "libertà", "liberdade"), ("war", "guerra", "guerre", "Krieg", "guerra", "guerra"),
    ("peace", "paz", "paix", "Frieden", "pace", "paz"), ("wisdom", "sabiduría", "sagesse", "Weisheit", "saggezza", "sabedoria"),
]
FRESH_YEARS = [("the French Revolution began", 1789), ("the Battle of Hastings was fought", 1066), ("the Magna Carta was sealed", 1215), ("Gutenberg printed his Bible", 1455),
               ("the Spanish Armada was defeated", 1588), ("the Great Fire of London broke out", 1666), ("the United States declared independence", 1776), ("the Battle of Waterloo was fought", 1815),
               ("the American Civil War began", 1861), ("the Suez Canal opened", 1869), ("the Eiffel Tower was completed", 1889), ("the Titanic sank", 1912),
               ("the Russian Revolution took place", 1917), ("the Wall Street Crash happened", 1929), ("the Second World War began", 1939), ("India gained independence", 1947),
               ("the first artificial satellite Sputnik was launched", 1957), ("the Cuban Missile Crisis occurred", 1962), ("Neil Armstrong walked on the Moon", 1969), ("the Chernobyl disaster happened", 1986),
               ("the Soviet Union was dissolved", 1991), ("the euro banknotes entered circulation", 2002), ("the first iPhone was released", 2007), ("the Sydney Olympic Games were held", 2000),
               ("the Berlin Wall was built", 1961), ("Queen Victoria came to the throne", 1837), ("the Boston Tea Party took place", 1773), ("the Wright brothers first flew", 1903)]
FRESH_ARITH = [(13, 13, "*"), (14, 14, "*"), (16, 16, "*"), (17, 17, "*"), (18, 18, "*"), (19, 19, "*"), (21, 21, "*"), (23, 3, "*"), (27, 4, "*"), (29, 3, "*"),
               (31, 4, "*"), (33, 5, "*"), (37, 3, "*"), (41, 4, "*"), (43, 5, "*"), (47, 3, "*"), (53, 4, "*"), (59, 3, "*"), (61, 5, "*"), (67, 3, "*"),
               (128, 3, "*"), (256, 3, "*"), (111, 7, "*"), (123, 5, "*"), (135, 4, "*"), (147, 3, "*"), (170, 5, "*"), (190, 6, "*"),
               (345, 678, "+"), (456, 789, "+"), (567, 891, "+"), (678, 912, "+"), (789, 123, "+"), (891, 234, "+"), (912, 345, "+"), (1234, 4321, "+"), (2345, 5432, "+"), (3456, 6543, "+")]
FRESH_NUMWORDS = [(16, 7, "+"), (17, 8, "+"), (18, 9, "+"), (19, 6, "+"), (21, 4, "+"), (23, 5, "+"), (24, 7, "+"), (26, 8, "+"), (27, 9, "+"), (28, 6, "+"),
                  (31, 7, "+"), (32, 9, "+"), (34, 8, "+"), (35, 6, "+"), (37, 7, "+"), (38, 9, "+"), (41, 8, "+"), (43, 6, "+"), (44, 9, "+"), (46, 7, "+"),
                  (47, 8, "+"), (49, 9, "+"), (51, 6, "+"), (52, 7, "+"), (53, 8, "+"), (56, 9, "+"), (57, 6, "+"), (58, 7, "+"), (61, 8, "+"), (62, 9, "+")]
FRESH_FORMULAS = [("sulfuric acid", "H2SO4"), ("hydrochloric acid", "HCl"), ("nitric acid", "HNO3"), ("sodium hydroxide", "NaOH"), ("potassium chloride", "KCl"),
                  ("calcium carbonate", "CaCO3"), ("sodium bicarbonate", "NaHCO3"), ("hydrogen peroxide", "H2O2"), ("nitrous oxide", "N2O"), ("sulfur dioxide", "SO2"),
                  ("ozone", "O3"), ("ethanol", "C2H5OH"), ("propane", "C3H8"), ("butane", "C4H10"), ("benzene", "C6H6"), ("acetylene", "C2H2"), ("ethylene", "C2H4"),
                  ("silicon dioxide", "SiO2"), ("magnesium oxide", "MgO"), ("calcium oxide", "CaO"), ("iron oxide (rust, ferric oxide)", "Fe2O3"), ("copper sulfate", "CuSO4"),
                  ("silver nitrate", "AgNO3"), ("potassium permanganate", "KMnO4"), ("sodium nitrate", "NaNO3"), ("phosphoric acid", "H3PO4"), ("hydrogen sulfide", "H2S"),
                  ("carbon monoxide", "CO"), ("nitrogen dioxide", "NO2"), ("lithium hydroxide", "LiOH")]


# ------------------------------------------------------------------ the disjointness filter -----------------------
def _word_in(needle, hay):
    return re.search(r"(?<![^\W_])" + re.escape(needle) + r"(?![^\W_])", hay) is not None


def _raw_item_strings():
    """the RAW (un-normalised) answer-side strings of the item files and bench banks, so a string's case is known."""
    out = set()
    for f in x3.ITEM_FILES:
        p = os.path.join(x3.DATA_DIR, f)
        if not os.path.exists(p): continue
        for it in json.load(open(p, encoding="utf-8")):
            out.update(x3._flat(it.get("string")) + x3._flat(it.get("target"))); out.add("".join(it.get("pieces_text", [])))
    bench = os.path.join(x3.DATA_DIR, "wsbench")
    for f in sorted(os.listdir(bench)) if os.path.isdir(bench) else []:
        if not f.endswith(".json"): continue
        for it in json.load(open(os.path.join(bench, f), encoding="utf-8")).get("items", []):
            out.update(x3._flat(it.get("target")) + x3._flat(it.get("target_alts")) + x3._flat(it.get("intermediates")) + x3._flat(it.get("typo_word")))
            for u in it.get("units") or []:
                for fs in (u.get("forms") or {}).values(): out.update(x3._flat(fs))
                out.update(x3._flat(u.get("match")))
            for b in it.get("bridges") or []: out.update(x3._flat(b.get("answers")))
    return {s for s in out if s and s.strip()}


def _proper(raw):
    """a name-like string: capitalised somewhere, or more than one word. Containment (both directions) is tested
    against these only; lowercase single words ('bird', 'black', 'water') are matched by equality only, so that
    'Larry Bird' or 'black pepper' are not banned by a common noun while 'Isaac Newton' bans 'Newton'."""
    s = raw.strip()
    return any(ch.isupper() for ch in s) or len(s.split()) > 1


def forbidden_strings():
    """(exact, proper): normalised strings a fresh item may not EQUAL (exact: every item-file / bench / table /
    carrier string) and the name-like subset it may not CONTAIN or BE CONTAINED IN as a whole word (proper)."""
    raw = _raw_item_strings()
    tabs = set()
    for row in mk.COUNTRIES + mk.PHRASE_COUNTRIES: tabs.update(str(x) for x in row if x and not isinstance(x, list)); tabs.update(y for x in row if isinstance(x, list) for y in x)
    for row in mk.PEOPLE: tabs.add(row[1])
    for row in mk.PHRASE_PEOPLE: tabs.add(row[1])
    for row in mk.ELEMENTS: tabs.add(row[1])
    for row in mk.ANIMALS: tabs.add(row[1])
    for row in mk.PHRASE_STATES: tabs.update(row[:3])
    for tbl in (mk.PHRASE_PLACES, mk.PHRASE_GEO, mk.PHRASE_EVENTS): tabs.update(r[1] for r in tbl)
    for d in (mk.PHRASE_CONCEPTS, mk.PHRASE_MORE):
        for rows in d.values(): tabs.update(r[1] for r in rows)
    for tbl in (mk.PHRASE_LANDMARK_BRIDGES, mk.PHRASE_CONCEPT_BRIDGES): tabs.update(r[1] for r in tbl)
    for r in mk.PHRASE_MORE_BRIDGES: tabs.add(r[2])
    for row in mk.TRANSLATIONS: tabs.update(str(x) for x in row if x)
    for _, f in mk.FORMULAS: tabs.add(f)
    for _, y in mk.YEARS: tabs.add(str(y))
    for a, b, op in mk.ARITH: tabs.add(str({"*": a * b, "+": a + b, "/": a // b}[op]))
    for a, b, op in mk.NUMWORD_PROBLEMS: tabs.add(mk.numword(a + b if op == "+" else a * b))
    raw |= {s for s in tabs if s}
    for kind, lst in x3.CARRIERS.items():
        for pr in lst: raw |= set(x3.exemplar_sides_of(pr))
    exact = {x3.norm(s) for s in raw if x3.norm(s)}
    proper = {x3.norm(s) for s in raw if _proper(s) and len(x3.norm(s)) >= 3}
    return exact, proper


def overlap_reason(string, forbidden):
    """None if the fresh string is entity-disjoint from forbidden = (exact, proper); else the offending string. A
    string overlaps when it EQUALS any forbidden string, or when it contains / is contained in a NAME-LIKE forbidden
    string (_proper: capitalised or multi-word, >= 3 characters) as a whole word / phrase. Digit strings are compared
    by equality only (contains-as-word would ban every number)."""
    exact, proper = forbidden
    n = x3.norm(string)
    if n in exact: return n
    if n.isdigit(): return None
    for f in proper:
        if f.isdigit(): continue
        if _word_in(f, n) or _word_in(n, f): return f
    return None


def build(tok, log=print):
    """the candidate items of both tracks, before and after the disjointness filter. Every word-phrase candidate is
    tried in MODE phrase first (track B); a candidate whose pieces are not all whole word tokens (' Fey' 'n' 'man')
    is a legitimate sub-word multi-token string and goes to track A under the SAME category, so both tracks get
    fresh person / landmark / event / compound items and their categories line up with the existing sets."""
    it = mk.item; add = mk.add; F = "fresh"
    out = {"B": [], "A": []}

    def route(cat, kind, prompt, string, target=None):
        mk.MODE = "phrase"; x = it(tok, F, cat, kind, prompt, string, target)
        if x is not None: add(out["B"], x); return
        mk.MODE = "multi"; add(out["A"], it(tok, F, cat, kind, prompt, string, target))

    for desc, name, nat, pcat in FRESH_PEOPLE + FRESH_PEOPLE_MORE:
        d0 = desc[0].lower() + desc[1:]
        route(pcat, "answer", f"Fact: {desc} was", " " + name)
        route(pcat, "answer", f"Fact: The name of {d0} is", " " + name)
        route("bridge_" + pcat + "_nationality", "bridge", f"Fact: The nationality of {d0} was", " " + name, nat)
    for pr, ph in FRESH_LANDMARKS: route("landmark", "answer", pr, mk.lead(pr) + ph)
    for pr, ph in FRESH_EVENTS: route("event", "answer", pr, mk.lead(pr) + ph)
    for pr, ph in FRESH_COMPOUNDS + FRESH_COMPOUNDS_MORE: route("compound", "answer", pr, mk.lead(pr) + ph)
    for pr, ph in FRESH_CHARACTERS: route("character", "answer", pr, mk.lead(pr) + ph)
    for pr, ph in FRESH_PLACES: route("place", "answer", pr, mk.lead(pr) + ph)
    for pr, ph, tg in FRESH_LANDMARK_BRIDGES: route("bridge_landmark_country", "bridge", pr, " " + ph, tg)
    cur_count = {}
    for c in FRESH_COUNTRIES: cur_count[c[2]] = cur_count.get(c[2], 0) + 1
    for country, cap, cur, lang, cont in FRESH_COUNTRIES:
        if cap: route("capital", "answer", f"Fact: The capital of {country} is", " " + cap)
        if cap and country.lower() not in cap.lower(): route("country_of_capital", "answer", f"Fact: {cap} is the capital of", " " + country)
        if cur and cur not in ("dollar", "euro", "franc"): route("currency", "answer", f"Fact: The unit of currency in {country} is the", " " + cur)
        if isinstance(lang, str): route("language", "answer", f"Fact: The official language of {country} is", " " + lang)
        if cap and country.lower() not in cap.lower():
            if cont: route("bridge_capital_continent", "bridge", f"Fact: The continent of the country whose capital is {cap} is", " " + country, cont)
            if cur and cur_count[cur] == 1: route("bridge_capital_currency", "bridge", f"Fact: The currency of the country whose capital is {cap} is the", " " + country, cur)
    mk.MODE = "multi"  # the rest is track A by construction (sub-word strings, digits, single-word answers)
    for desc, name, nat, pcat in FRESH_PEOPLE + FRESH_PEOPLE_MORE:
        w = name.split(); d0 = desc[0].lower() + desc[1:]
        if len(w) >= 2 and w[-1][0].isupper() and w[-2].lower() not in ("the", "of", "van", "da", "de", "von", "queen") and w[0] not in ("Alexander", "Catherine", "Peter", "William", "Joan", "Mary", "Henry", "Queen", "Prince", "Princess", "King", "Kim"):
            add(out["A"], it(tok, F, "surname", "answer", f"Fact: The surname of {d0} is", " " + w[-1]))
            add(out["A"], it(tok, F, "bridge_person_nationality", "bridge", f"Fact: The nationality of {d0} was", " " + w[-1], target=nat))
    for sym, name, z in FRESH_ELEMENTS:
        add(out["A"], it(tok, F, "element", "answer", f"Fact: The chemical element with the symbol {sym} is", " " + name))
        add(out["A"], it(tok, F, "bridge_element_number", "bridge", f"Fact: The atomic number of the element with the symbol {sym} is", " " + name, target=str(z)))
    for pr, animal in FRESH_ANIMALS: route("animal", "answer", "Fact: " + pr, " " + animal)
    for row in FRESH_WORDS:
        for lname, j in mk.LANGS:
            if row[j]: add(out["A"], it(tok, F, f"translation_{lname.lower()}", "answer", f'Fact: The {lname} word for "{row[0]}" is "', row[j]))
    for a, b, op in FRESH_NUMWORDS:
        r = a + b if op == "+" else a * b
        add(out["A"], it(tok, F, "number_word", "answer", f"Fact: {mk.numword(a).capitalize()} {'plus' if op == '+' else 'times'} {mk.numword(b)} equals", " " + mk.numword(r)))
    for name, f in FRESH_FORMULAS: add(out["A"], it(tok, F, "formula", "answer", f"Fact: The chemical formula of {name} is", " " + f))
    for ev, y in FRESH_YEARS: add(out["A"], it(tok, F, "year", "answer", f"Fact: {ev[0].upper() + ev[1:]} in the year ", str(y)))
    for a, b, op in FRESH_ARITH:
        r = {"*": a * b, "+": a + b, "/": a // b}[op]
        add(out["A"], it(tok, F, "arith_digits", "answer", f"{a} {op} {b} = ", str(r)))
    # the disjointness filter (both tracks): drop and record
    forb = forbidden_strings(); kept, dropped = {"B": [], "A": []}, []
    for tr in ("B", "A"):
        for x in out[tr]:
            why = overlap_reason(x["string"], forb)
            if why is None: kept[tr].append(x)
            else: dropped.append({"track": tr, "id": x["id"], "string": x["string"], "category": x["category"], "kind": x["kind"], "overlaps": why})
    for tr in ("B", "A"):
        kept[tr], ndup = mk.dedupe_cap(kept[tr], cap=0, seed=0, cap_string=3)
        for x in kept[tr]: x["track"] = tr
    log(f"fresh items: track B {len(kept['B'])} ({sum(x['kind'] == 'answer' for x in kept['B'])} answer / {sum(x['kind'] == 'bridge' for x in kept['B'])} bridge), track A {len(kept['A'])} ({sum(x['kind'] == 'answer' for x in kept['A'])} answer / {sum(x['kind'] == 'bridge' for x in kept['A'])} bridge); dropped for entity overlap {len(dropped)}")
    return kept, dropped, forb


def check_items(items, forb):
    """the contract every fresh item must satisfy (also run by tests/test_wave5_tiny.py): tokenisation round-trips,
    >= 2 pieces, first piece alphanumeric, track-B pieces are whole space-prefixed word tokens, the string is not in
    the prompt, and the string is entity-disjoint from `forb`."""
    bad = []
    for x in items:
        p = x["pieces_text"]
        if len(p) < 2: bad.append((x["id"], "pieces < 2"))
        if "".join(p) != x["string"]: bad.append((x["id"], "no round trip"))
        if not any(ch.isalnum() for ch in p[0]): bad.append((x["id"], "first piece not alphanumeric"))
        if x["track"] == "B" and not (mk._is_word_piece(p[0], lead_space=x["string"].startswith(" ")) and all(mk._is_word_piece(t) for t in p[1:])): bad.append((x["id"], "track B piece is not a whole word token"))
        if x["string"].strip().lower() in x["prompt"].lower(): bad.append((x["id"], "string in prompt"))
        if overlap_reason(x["string"], forb) is not None: bad.append((x["id"], "entity overlap: " + str(overlap_reason(x["string"], forb))))
        if x["kind"] == "bridge" and not x.get("target"): bad.append((x["id"], "bridge without target"))
    return bad


def main():
    p = argparse.ArgumentParser(); p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B", help="the Qwen3 tokenizer (shared by 1.7B / 8B / 14B)")
    p.add_argument("--out", default=os.path.join(ROOT, "data", "fresh_items.json")); p.add_argument("--check", action="store_true", help="build and report, write nothing"); a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    kept, dropped, forb = build(tok)
    items = kept["B"] + kept["A"]; bad = check_items(items, forb)
    assert not bad, f"{len(bad)} contract violations: {bad[:10]}"
    c = mk.counts(items)
    for k in sorted(c): print(f"  {k[0]} {k[1]:6s} {k[2]:36s} {c[k]}")
    print(f"dropped ({len(dropped)}): " + ", ".join(f"{d['string'].strip()!r}~{d['overlaps']!r}" for d in dropped[:40]) + (" ..." if len(dropped) > 40 else ""))
    if a.check: return
    json.dump(items, open(a.out, "w"), indent=1, ensure_ascii=False)
    split = {"seed": 0, "rule": "every kind=='answer' id is HELD-OUT and none is tuning: nothing is chosen on this set (prereg A24); bridges are outside the split as in h2a_split.json",
             "tuning": [], "heldout": sorted(x["id"] for x in items if x["kind"] == "answer")}
    json.dump(split, open(os.path.join(os.path.dirname(a.out), "fresh_split.json"), "w"), indent=1)
    json.dump(dropped, open(os.path.join(os.path.dirname(a.out), "fresh_items_dropped.json"), "w"), indent=1, ensure_ascii=False)
    print(f"wrote {a.out} ({len(items)} items), fresh_split.json ({len(split['heldout'])} held-out answer ids), fresh_items_dropped.json ({len(dropped)})")


if __name__ == "__main__":
    main()
