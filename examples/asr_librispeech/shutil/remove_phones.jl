lexi=ARGS[1]
lexi_out=ARGS[2]

function main(lexicon)
  # remove phones 
  L = read(lexi, String);
  D = Dict()
  D["AA0"] = D["AA1"] = D["AA2"] = "AA"
  D["AE0"] = D["AE1"] = D["AE2"] = "AE"
  D["AH0"] = D["AH1"] = D["AH2"] = "AH"
  D["AO0"] = D["AO1"] = D["AO2"] = "AO"
  D["AW0"] = D["AW1"] = D["AW2"] = "AW"
  D["AY0"] = D["AY1"] = D["AY2"] = "AY"
  D["EH0"] = D["EH1"] = D["EH2"] = "EH"
  D["ER0"] = D["ER1"] = D["ER2"] = "ER"
  D["EY0"] = D["EY1"] = D["EY2"] = "EY"
  D["IH0"] = D["IH1"] = D["IH2"] = "IH"
  D["IY0"] = D["IY1"] = D["IY2"] = "IY"
  D["OW0"] = D["OW1"] = D["OW2"] = "OW"
  D["OY0"] = D["OY1"] = D["OY2"] = "OY"
  D["SH0"] = D["SH1"] = D["SH2"] = "SH"
  D["UH0"] = D["UH1"] = D["UH2"] = "UH"
  D["UW0"] = D["UW1"] = D["UW2"] = "UW"

  for d in D
    L = replace(L,d)
  end
  return L
end

L = main(lexi)
write(lexi_out, L)
