1. Utvärdera whisper small, medium och large-v3 på ett dataset innehållandes entiteter
2. Träna filtreringsmodellen
3. Skapa upp det syntetiska datasettet utifrån genererad text eller ett färdigt dataset
4. Filtrera bort dåliga samples 
5. Fine-tunea whisper modellerna på olika nivåer av filtrerade dataset
6. Utvärdera de nya varianterna av whisper small, medium och large-v3 på ett dataset innehållandes entiteter


# Filtering framework
Gick inte bra med BERT encoder och whisper medium från kb, alla fördelade runt 0.5

Idéer framåt: 
Byt encoder / encoders
Frys inte encodern
Refinea loss funktionen
Höja / byta learning rate?


# Data generering
Slut på elevenlabs credits
Vet inte vad jag ska ha för text underlag
Facebook fungerade ganska bra på sagemaker notebooken, 100 samples tog typ en minut
Dåliga speakers: 2, 3, 4, 9
Ok speakers: 1, 5, 6, 7, 8

Hög temperatur
Structured output

Gick inte bra - (
Lång lista av bolag / namn / platser - Tänk på om det är värt att ha med metadata om företagen 
Generera en mening med spontantal innahållande.
Flera meningar upp till 30 sekunder )

- Byt ut entiteten {x} i denna mening till {y}
# Whisper finetuning


# Utvärdering
KBlabs whisper vs Fine tunad KBLabs whisper
Första gången på ett syntetiskt dataset
Exakt strängmatchning / position? 
Blir det inte skevt att utvärdera whisper på ett syntetiskt dataset och jämföra med samma dataset efter?
Referensdatasetet måste ju nästan vara riktigt ljud

1. En idé är att generera syntetiskt ljud med elevenlabs och använda 80 % för träning och 20% för utvärdering
- Försöka vikta detta för att träningsdatan och utvärderingen skall innehålla samma entiteter men olika meningar

Går det att utvärdera ENTITETER på NST? I och med att dem bara har utvärderat på WER.
- Ett extra steg som kan bli fel - Extraherade entiteter kan vara fel
- Återigen problemet med text-generering

Annars 
- Hitta ett dataset med riktigt ljud som innehåller entiteter som whisper inte är tränad på
- Utvärdera på dem entiteterna 
- Syntetisera meningar med dem entiteterna 


Pipeline: 
1. Kör NST, Commonvoice och Fleurs igenom KBLabs entitetsmodell, behåll endast de med entiteter // Evaluerar KB på test eller allt? Farhåga: Fel entiteter ✅
2. Evaluera på entity score✅
3. Behåll en lista på entiterna som de ej sätter. ✅
4. Gå igenom nyhets-korpuset med GPT och listan - Byt ut entiteten {x} i denna mening till {y}, möjligt att man behöver gruppera efter entitetstyp ✅
5. Skapa ljud via elevenlabs
6. Dra detta genom filtreringsmodellen
7. Ta bort dåliga syntetiska ljud och text par
8. Fine-tunea whisper på de nya meningarna
9. Kör evalueringen igen 


Att få gjort idag: ✅
Få till felfri entity extraction från CV och Fluers ✅
Kör igenom hela dataseten och lägg upp på huggingface ✅
Rätta eventuella fel i entiteterna genom en LLM ✅
Evaluera whisper på dataseten och få ut en lista på alla entiteter som inte sitter med tillhörande entitetstyp ✅
Få till scriptet som byter ut entiteterna i svt-meningarna ✅
Starta syntetiseringen av den nya svt datan // Intressanta är hur meningar man genererar per entitet

Att få gjort den här veckan:
Fine-tunea whisper på det nya datasetet
Utvärdera på entiteterna från commonvoice och fleurs

Något att tänka på: Hur WER och CER påverkas av nya entiteterna
Möjlig förklaring, problem med kontext 

# Lärdomar:
Det är krångligt att få ihop ett rimligt text-dataset 
Om man skall göra detta för domain adaptation och riktade insatser fungerar det bäst om det finns stora mängder text
Rimligtvis så hade man scrapeat text-meningarna om företag etc.



# Sista TODO
Skriv om introduktionen för att passa vad experimentet blev ✅
Skriv om metodens validitet och reliabilitet ✅
Skriv mer ingående om resultat
Skriv conclusion & future work kapitlet
Skriv abstrakt och subtitel
Skriv keywords
Läs igenom 100 gånger 
Klart! 

Kap 1 Introduktion:
Kap 2 Bakgrund:
Kap 3 Experiment design:
Kap 4 Implementation:
Kap 5 Resultat:
Kap 6 Future work:
Kap 7 Referenser: ✅
