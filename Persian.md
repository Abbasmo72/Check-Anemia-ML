<div align="center">

## تشخیص کم خونی
<img src="https://i.pinimg.com/originals/a8/f2/84/a8f284cf430e9b5ee5df1d02395ba11c.gif" height="200px" width="500px">
</div>
<hr>

### کم خونی: علائم و عوارض
کم خونی یک وضعیت پزشکی است که با تعداد کمتر از حد طبیعی گلبول های قرمز یا سطح هموگلوبین در خون مشخص می شود. این عارضه می تواند به دلایل مختلفی از جمله کمبود آهن، کمبود ویتامین B12 یا فولات، اختلالات ژنتیکی و بیماری های مزمن ایجاد شود. علائم کم خونی ممکن است شامل خستگی شدید، رنگ پریدگی پوست، سرگیجه، تنگی نفس و ضربان قلب سریع باشد.
عوارض کم خونی می تواند جدی باشد، به خصوص اگر درمان نشود. این عوارض شامل خطر بیشتر عفونت ها، مشکلات قلبی عروقی مانند نارسایی قلبی و کاهش توانایی بدن برای تحمل فعالیت های فیزیکی است. در موارد شدید، کم خونی می تواند منجر به اختلالات جدی تری مانند سکته مغزی یا حمله قلبی شود.
تشخیص کم خونی معمولاً از طریق آزمایش خون انجام می شود و درمان بستگی به نوع کم خونی دارد. برای کمبود آهن، مکمل های آهن و تغییرات رژیم غذایی ممکن است موثر باشد. در صورت کمبود ویتامین، مکمل های مربوطه تجویز می شود.

## رگرسیون لجستیک و عملکرد آن
رگرسیون لجستیک یک مدل آماری است که برای پیش بینی احتمال وقوع یک رویداد خاص استفاده می شود. این مدل کاربردهای گسترده ای در زمینه هایی مانند پزشکی، مالی و علوم اجتماعی دارد. برخلاف رگرسیون خطی که برای پیش‌بینی مقادیر عددی استفاده می‌شود، رگرسیون لجستیک برای پیش‌بینی متغیرهای باینری (دو گزینه‌ای) طراحی شده است.
عملکرد رگرسیون لجستیک با استفاده از تابع لجستیک برای مدل‌سازی احتمال وقوع یک رویداد کار می‌کند. این تابع مقادیر پیش بینی شده را بین 0 و 1 نگه می دارد و برای پیش بینی احتمالات مناسب است. در این مدل، ضرایب به دست آمده از داده ها به پیش بینی کمک می کند.
برای ارزیابی عملکرد یک مدل رگرسیون لجستیک، معمولاً از معیارهایی مانند دقت، یادآوری و دقت استفاده می‌شود. این معیارها به ما کمک می‌کنند تا بفهمیم مدل چقدر خوب عمل می‌کند و چه نسبتی از پیش‌بینی‌ها درست است.
رگرسیون لجستیک می تواند بینش ارزشمندی در مورد اهمیت هر ویژگی در پیش بینی نتیجه ارائه دهد. این اطلاعات می تواند در تصمیم گیری و بهینه سازی استراتژی ها در حوزه های مختلف بسیار سودمند باشد.

## تجزیه و تحلیل و پیش بینی کم خونی با استفاده از رگرسیون لجستیک
این کد یک مدل رگرسیون لجستیک را برای پیش بینی کم خونی بر اساس داده های آزمایش خون ایجاد و ارزیابی می کند. در اینجا مراحل مربوط به کد توضیح داده شده است:
1. <b>وارد کردن کتابخانه ها:</b>
  - پانداها: برای دستکاری داده ها.
  - sklearn.model_selection: برای تقسیم داده ها به مجموعه های آموزشی و آزمایشی.
  - sklearn.linear_model: برای استفاده از مدل رگرسیون لجستیک.
  - sklearn.metrics: برای محاسبه دقت مدل.
2. <b>بارگیری و آماده سازی مجموعه داده:</b>
  - داده ها از یک فایل CSV با نام CBCdata_for_meandeley_csv.csv بارگیری می شوند.
  - نام ستون ها به صورت دستی تنظیم می شود.
  - ردیف اول حاوی توضیحات حذف می شود.
  - داده ها به انواع عددی تبدیل می شوند و هر ردیفی که مقادیر گم شده باشد حذف می شود.
3. <b>تعریف ویژگی های ورودی (X) و متغیر هدف (y):</b>
  - ویژگی ها شامل مقادیر حاصل از آزمایش خون (RBC، PCV، MCV، MCH، MCHC، RDW، TLC، PLT) است.
  - متغیر هدف نشان می دهد که آیا هموگلوبین (HGB) کمتر از 12 است یا خیر. اگر کمتر باشد، 1 (کم خونی) و در غیر این صورت 0 (بدون کم خونی) مشخص می شود.
4. <b>تقسیم داده ها:</b>
  - داده ها به 80 درصد برای آموزش و 20 درصد برای آزمایش تقسیم می شوند.
5. <b>آموزش مدل:</b>
  - یک مدل رگرسیون لجستیک ایجاد شده و بر روی داده های آموزشی آموزش داده می شود.
6. <b>ارزیابی مدل:</b>
  - پیش بینی هایی برای مجموعه تست انجام می شود و دقت مدل محاسبه و چاپ می شود.
7. <b>پیش‌بینی با ورودی کاربر:</b>
  - برنامه از کاربر می خواهد مقادیری را برای ویژگی ها وارد کند.
  - پیش بینی بر اساس ورودی کاربر انجام می شود و نتیجه (احتمال یا عدم احتمال کم خونی) نمایش داده می شود.

## در اینجا تجزیه و تحلیل خط به خط کد ارائه شده است:
1. این خط کتابخانه پانداها را وارد می کند که برای دستکاری و تجزیه و تحلیل داده ها استفاده می شود.
```python
import pandas as pd
```
2. این خط تابع train_test_split را از ماژول sklearn.model_selection وارد می کند، که برای تقسیم مجموعه داده به مجموعه های آموزشی و آزمایشی استفاده می شود.
```python
from sklearn.model_selection import train_test_split
```
3. این خط مدل LogisticRegression را از ماژول sklearn.linear_model وارد می کند که برای ایجاد مدل رگرسیون لجستیک استفاده خواهد شد.
```python
from sklearn.linear_model import LogisticRegression
```
4. این خط تابع accuracy_score را از ماژول sklearn.metrics وارد می کند که برای محاسبه دقت مدل استفاده می شود.
```python
from sklearn.metrics import accuracy_score
```
5. این خط داده ها را از یک فایل CSV به نام CBCdata_for_meandeley_csv.csv بارگیری می کند و آن را در یک DataFrame به نام داده ذخیره می کند.
```python
data = pd.read_csv('CBCdata_for_meandeley_csv.csv')
```
6. این خط به صورت دستی نام ستون های DataFrame را تنظیم می کند.
```python
data.columns = ['S.No', 'Age', 'Sex', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT', 'HGB']
```
7. ین خط اولین ردیف DataFrame را که معمولاً حاوی توضیحات یا هدر است حذف می کند.
```python
data = data[1:]
```
8. این خط سعی می کند تمام مقادیر موجود در DataFrame را به انواع عددی تبدیل کند. اگر هر گونه خطایی رخ دهد (مانند مقادیر غیر عددی)، آن مقادیر به NaN تبدیل می شوند.
```python
data = data.apply(pd.to_numeric, errors='coerce')
```
9. این خط تمام ردیف های حاوی مقادیر گمشده (NaN) را از DataFrame حذف می کند.
```python
data = data.dropna()
```
10. این خط ویژگی های ورودی (X) را مشخص می کند که شامل مقادیر آزمایش های مختلف خون می شود.
```python
X = data[['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT']]
```
11. این خط متغیر هدف (y) را تعریف می کند. اگر سطح هموگلوبین (HGB) کمتر از 12 باشد، مقدار را 1 (نشان دهنده کم خونی) و در غیر این صورت 0 (نشان دهنده کم خونی) تنظیم می کند.
```python
y = (data['HGB'] < 12).astype(int)
```
12. این خط داده ها را به مجموعه های آموزشی و آزمایشی تقسیم می کند. 80 درصد از داده ها را برای آموزش و 20 درصد را برای آزمایش اختصاص می دهد. random_state=42 تضمین می‌کند که هر بار که کد اجرا می‌شود، داده‌ها به یک شکل تقسیم می‌شوند.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
13. این خط یک نمونه جدید از مدل رگرسیون لجستیک ایجاد می کند.
```python
model = LogisticRegression()
```
14. این خط با استفاده از داده های آموزشی، مدل رگرسیون لجستیک را آموزش می دهد.
```python
model.fit(X_train, y_train)
```
15. این خط پیش بینی هایی را روی مجموعه تست انجام می دهد و نتایج را در y_pred ذخیره می کند.
```python
y_pred = model.predict(X_test)
```
16. این خط دقت مدل را با مقایسه مقادیر پیش بینی شده با مقادیر واقعی محاسبه می کند.
```python
accuracy = accuracy_score(y_test, y_pred)
```
17. این خط دقت مدل فرمت شده به دو رقم اعشار را چاپ می کند.
```python
print(f"Model accuracy: {accuracy:.2f}")
```
18. این خط دقت مدل فرمت شده به دو رقم اعشار را چاپ می کند. این خط از کاربر می خواهد مقادیر ورودی ویژگی ها را وارد کند و این مقادیر را در لیستی به نام user_input ذخیره می کند. هر ورودی به یک شناور تبدیل می شود.
```python
user_input = [float(input(f"Please enter the value for {col}: ")) for col in X.columns]
```
19. این خط بر اساس ورودی کاربر پیش بینی می کند و نتیجه را در پیش بینی ذخیره می کند.
```python
prediction = model.predict([user_input])
```
20. این بخش نتیجه پیش‌بینی را بررسی می‌کند و یک پیام مناسب به کاربر نشان می‌دهد: اگر پیش‌بینی 1 باشد، نشان‌دهنده احتمال کم خونی است. اگر 0 باشد نشان دهنده کم خونی است.
```python
if prediction[0] == 1:
    print("Result: Anemia is likely.")
else:
    print("Result: No anemia.")
```


## کد پایتون
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and prepare the dataset
data = pd.read_csv('CBCdata_for_meandeley_csv.csv')
data.columns = ['S.No', 'Age', 'Sex', 'RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT', 'HGB']

# Remove the first row containing descriptions
data = data[1:]

# Convert data to numeric and drop any rows with missing values
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# Define input features (X) and target variable (y)
X = data[['RBC', 'PCV', 'MCV', 'MCH', 'MCHC', 'RDW', 'TLC', 'PLT']]
y = (data['HGB'] < 12).astype(int)  # Anemia is indicated if Hemoglobin is less than 12

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Prediction with new user input
user_input = [float(input(f"Please enter the value for {col}: ")) for col in X.columns]
prediction = model.predict([user_input])

# Display result based on prediction
if prediction[0] == 1:
    print("Result: Anemia is likely.")
else:
    print("Result: No anemia.")
```

