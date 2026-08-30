from langchain_community.document_loaders import WebBaseLoader

url = "https://www.amazon.in/Happi-Planet-Multi-Surface-Cleaning-Bathroom/dp/B0GPN9R146/ref=zg_bs_c_home-improvement_d_sccl_1/522-1496555-8888053?pd_rd_w=p7Pyb&content-id=amzn1.sym.b908f532-cbe7-4274-8b24-b671acc58bd2&pf_rd_p=b908f532-cbe7-4274-8b24-b671acc58bd2&pf_rd_r=MTC69W5079TB17DYXMW3&pd_rd_wg=Efrcn&pd_rd_r=e044ba5a-877d-4768-b24c-1922f143ab92&pd_rd_i=B0GPN9R146&th=1"
loader =  WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)