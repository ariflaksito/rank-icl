import re
import argparse
import wikipedia
from cleantext import clean

parser = argparse.ArgumentParser(description="use --help for investigating input params")

parser.add_argument('--output', type=str, required=True, help="full path of output file (make sure using .csv)")
parser.add_argument('--queries', type=str, required=True, help="full path of lqueries file")

args = parser.parse_args()

fileout = args.output
filename = args.queries

ignored = ['See also', 'References', 'Further reading', 'External links', 'Sources']

def extract_sections(text):
    # Regular expression to match sections and their content
    pattern = re.compile(r'\n={2}(?P<header>.+?)\s={2}\n*(.*?)\s*(?==|$)', re.DOTALL)
    matches = pattern.findall(text)
    
    # Clean up the matched data
    sections = [{'title': match[0].strip(), 'text': match[1].strip()} for match in matches]
    return sections

def remove_quotes(text):
    # Remove single and double quotes from the text
    return text.replace('"', '').replace("'", '')

def word_count(text):
    return len(text.split())

n_true = 0
n_false = 0
n_query = 0

with open(fileout, 'a', encoding='utf-8') as out_f:
    for line in open(filename, 'r'):
        n_query = n_query+1
        try:
            print(f'Processing {n_query}: {line.strip()}')
            page = wikipedia.page(line)
            page.content.encode('utf8')
            text = page.content

            x = extract_sections(text)
            for section in x:
                sentences = clean(section['text'], 
                                  fix_unicode=True, 
                                  to_ascii=True, 
                                  no_line_breaks=True,
                                  lower=False,
                                  no_punct=True,
                                  lang='en')
                explanation = remove_quotes(section['title'])
                
                if explanation in ignored:
                    continue
                
                if word_count(sentences) < 128 or word_count(sentences) > 512:
                    continue
                
                out_f.write(f'"{line.strip()}", "{explanation}", "{sentences}"\n')
                print(f'Save to file..')
                n_true = n_true+1
            
        except Exception as e:
            print(f'Error extract {line.strip()}')
            n_false = n_false+1
            continue 

print('\n\n ----------- Resume ---------\n')
print(f'Total query: {n_query}')
print(f'Number of Section: {n_true+n_false}')
print(f'Number of Success: {n_true}')
print(f'Number of Error: {n_false}')