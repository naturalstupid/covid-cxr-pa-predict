import pandas as pd
import shutil
url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/"
url1 = "https://github.com/ieee8023/covid-chestxray-dataset/raw/master/"
url_csv_file = url + "metadata.csv"

metadata = pd.read_csv(url_csv_file) #('./covid-chestxray-dataset/metadata.csv')
finding = 'COVID-19'
cxr_view = 'PA'
images = metadata[(metadata['finding'] == finding) & (metadata['view'] == cxr_view)]
print (len(images)," images found with (metadata['finding'] == " + finding +") & (metadata['view'] == 'PA')")
exit()
images.reset_index()

for _, row in images.iterrows():
    remote_file = url1 + f"images/{row.filename}"
    local_file = f"../data/covid-19/{row.filename}"
    print("copying "+remote_file+" to " + local_file)
    shutil.copy(remote_file, local_file)
