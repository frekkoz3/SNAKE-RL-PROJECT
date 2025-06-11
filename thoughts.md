Ho provato vari binning e lascio qui i risultati:
  1) FALLITO
     Ho provato a dargli il suo intorno ma non la posizione del cibo.
     Non funziona perchè i rewards sono troppo sparsi e lui preferisce o uccidersi (se la penalità per step è alta) o andare in loop infinito
  2) DA CONTINUARE
     Ho provato a dargli direzione del cibo + VN con distanza 2. 
     Il numero di stati è abbastanza più alto (circa 500 invece di 32). 
     Andrebbe fatto un allenamento molto lungo, e poi vedere se si comporta meglio. La sensazione è che non merita 
  3) DA CONTINUARE
     Ho provato VN1 + FP + lunghezza della coda.
     L'idea era che lui cambiasse comportamento con la coda lunga (magari movimenti più ampi per non incastrarsi)
     Il problema è che come la coda si allunga lui perde ciò che aveva imparato. Quindi serve un training più lungo
  4) FALLITO
     Ho dato solo le 4 direzioni cardinali, invece anche di quelle intermedie(NE, SW, ...) e non so perchè ma non impara nulla. Credo non sono         abbastanza informazioni
