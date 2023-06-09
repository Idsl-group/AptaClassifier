# Import scrapy
import scrapy
# Import the CrawlerProcess: for running the spider
from scrapy.crawler import CrawlerProcess
# Import scrapy
import scrapy
# Import the CrawlerProcess: for running the spider
from scrapy.crawler import CrawlerProcess


# Create the Spider class
class NDB_Spider(scrapy.Spider):
  name = "ndb_spider"
  # start_requests method
  def start_requests(self):
    urls = 'http://ndbserver.rutgers.edu/service/ndb/atlas/gallery/dna?polType=all&protFunc=all&strGalType=dna&stFeature=all&expMeth=all&galType=table&start=0&limit=8780'
    yield scrapy.Request(url = urls, callback = self.parse_front)

  #Aplicando los parametros de seleccion
  def parse(self, response):
    yield scrapy.FormRequest.from_response(response=response,formdata={'strGalType': 'dna', 'stFeature': 'single'},callback=self.parse_front)

 # Parsing the front page
  def parse_front(self, response):
    dna_blocks = response.css('a.rtAtlasLinkSmall')
    dna_links = dna_blocks.xpath('./@href[1]')
    links_to_follow = dna_links.extract()
    for url in links_to_follow:
         yield response.follow(url = url,callback = self.parse_pages)   

  def parse_pages(self, response):
    if "rcsb" in response.url:
      return
    Sequence_dirty=response.css('p.chain *::text')
    Sequence=Sequence_dirty.extract_first().strip()
    Target= 'Protein'
    Nucleotide= 'DNA'
    NDB_dict[tuple(Sequence)] =Target,Nucleotide
    

# Initialize the dictionary **outside** of the Spider class
NDB_dict = dict()

# Run the Spider
process = CrawlerProcess()
process.crawl(NDB_Spider)
process.start()    


for keys, values in NDB_dict.items():
    print(keys, values)