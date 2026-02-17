import requests 
from bs4 import BeautifulSoup
import pandas as pd 


url = 'https://thaimooc.ac.th/'


class thaimooc_scraper:
    def __init__(self, url):
        self.url = url
        self.html = ""
        self.category = ['คอมพิวเตอร์และเทคโนโลยี','ธุรกิจและการบริหารจัดการ','สุขภาพและการแพทย์','ภาษาและการสื่อสาร']
        self.category_urls = {}

        self.data = pd.DataFrame(columns=['Category','Course Description'])

    def fetch_page_content(self):
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            self.html = response.text
            print("Page content fetched successfully✅.")
        except Exception as e:
            print(f"Error fetching page content❌: {e}")
            return ""
    
    def parse_category_urls(self):
        soup = BeautifulSoup(self.html, "html.parser")

        category_urls = {key:"" for key in self.category}

        try:
            target_ul = soup.find("ul", {"class": "apus-vertical-menu nav"})
        
            print('Target UL found✅.')

            for a in target_ul.find_all("a", href=True):
                category_name = a.text.strip()
                if category_name in self.category:
                    category_urls[category_name] = a['href']

            self.category_urls = category_urls
            print("Category URLs parsed successfully✅.")
        except Exception as e:
            print(f"Error parsing category URLs❌: {e}")
            self.category_urls = {key:"" for key in self.category}

    def _parse_course_urls(self,course_url)->list[str]:

        try:
            courses_urls = []
            response = requests.get(course_url)
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            course_links = soup.find_all("div",{"class": "col-lg-4 col-md-6 col-12"}) #adjust

            for div in course_links:
                a_tag = div.find("a",href=True)
                if a_tag:
                    courses_urls.append(a_tag['href'])
                    print(f"Course URL found: {a_tag['href']}")
                    print(f"Course count: {len(courses_urls)}.")
        
            return courses_urls

        except Exception as e:
            print(f"Error parsing course URLs❌: {e}")
            return []
        
    def _parse_course_description(self, course_url)->str:
        try:
            response = requests.get(course_url)
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, "html.parser")

            description_div = soup.find("div", {"class": "course-tabs-scrollspy"})
            
            course_description = description_div.find('p').text.strip() 
            print("Course description parsed successfully✅.")
            return course_description

        except Exception as e:
            print(f"Error parsing course description❌: {e}")
            return ""
        
    def run(self):
        self.fetch_page_content()
        self.parse_category_urls()
        print(self.category_urls)

        for category, url in self.category_urls.items():
            print(f"Processing category: {category}")
            course_urls = self._parse_course_urls(url)
            print(f"Total course URLs found: {len(course_urls)}.")
            for course_url in course_urls:
                course_description = self._parse_course_description(course_url)

                row = {'Category': category, 'Course Description': course_description}

                self.data.loc[len(self.data)] = row

                print(f"Course description added to DataFrame✅. Current row count: {len(self.data)}.")
        
        print("Scraping completed✅.")
        self.data.to_csv('thaimooc_courses.csv', index=False, encoding='utf-8-sig')
        return self.data
            
scraper = thaimooc_scraper(url)
scraper.run()






