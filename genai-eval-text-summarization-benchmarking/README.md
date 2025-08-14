$${\color{red}This \space sample \space  was \space  archived \space  by \space  the \space  owner \space  on \space  August \space  14th, \space  2025. \space  It \space  is \space  now \space  read-only.}$$


# Qualitative Assessment of Text Summarizations
Perform a qualitative assessment of a candidate summarization by comparing it to a reference response. Metrics calculated: 

__BLEU:__ Measures the overlap of n-grams between a candidate text and reference text, emphasizing precision and brevity in translation tasks.

__ROUGE-N:__ Evaluates the overlap of n-grams between candidate and reference texts, focusing on recall to assess content similarity in summarization tasks.

__BERTScore:__ Uses contextual embeddings to compare semantic similarity between candidate and reference texts, capturing meaning beyond exact matches.

## Installation
Get started by installing conda and setting up your python enviornment.
```
./install.sh
```

Note: if this script has already been performed and you'd like to re-install the sample project only, the following command can be used to skip the re-install of dependencies.
```
./install.sh --skip
```

## Run Examples
The following command will compare the file "candidate_summarization.txt" against the file "reference_summarization.txt" and generate the above metrics. 
```
./run-demo.sh
```
