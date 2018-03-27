def input_date():
    year = int(input("__년도를 입력하시오:"))
    month = int(input("__월을 입력하시오:"))
    day = int(input("__일을 입력하시오:"))

    return year,month,day

def get_day_name(year,month,day):
    list=[0,31,28,31,30,31,30,31,31,30,31,30,31]
    listy = [0,31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    Dw=['일요일','월요일','화요일','수요일','목요일','금요일','토요일']
    year1 = year-1   #올해 미포함
    n = year1//4
    m=year1//100
    q=year1//400
    Ynum=n-m+q   #윤년의 갯수
    YYnum = year - Ynum-1   #윤년이 아닌해의 갯수, 올해 제외
    if year%4==0:  #윤년과 윤년이 아닌해의 리스트 골라주기
        li=listy
        if year % 100 == 0:
            li = list
            if year % 400 == 0:
                li = listy
            else:
                li = list
        else:
            li = listy
    else:
        li = list
    total_day = (Ynum * 366) + (YYnum * 365) + (sum(li[0:month])) + day #총일수계산
    day_num=(total_day%7)
    day_name = Dw[day_num]#요일계산

    return day_name

def is_leap(year):
    if year%4==0:#윤년과 윤년이 아닌해의 골라주기
        a=True
        if year % 100 == 0:
            a = False
            if year % 400 == 0:
                a = True
            else:
                a = False
        else:
            a = True
    else:
        a = False
    return a

if __name__ == '__main__':
    year, month, day = input_date()
    day_name = get_day_name(year,month,day)
    print(day_name)
    if is_leap(year) == True:
        print('입력하신 %s은 윤년입니다.'% year)