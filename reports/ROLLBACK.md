# Откат версии данных через DVC

Проверяем, что `git checkout HEAD~1 && dvc checkout` возвращает старую версию
данных.

| Коммит | Сообщение | md5 `data/raw/churn.csv` | Строк (с заголовком) |
|---|---|---|---|
| `c7a7f95` (HEAD) | return dataset to 20000 | `a0038705ffe95c3e1856f5f58bb0cbff` | 20001 |
| `1cfd6f8` (HEAD~1) | add minio | `2f7172b1cb306bf9794dd5d4db56cf2a` | 5001 |

Git хранит только `data/raw/churn.csv.dvc` с md5, а `dvc checkout` достаёт из
кэша файл с этим md5.

## Вывод терминала

```
(.venv) juliamikhailova@Julias-MacBook-Pro mlops-course % wc -l data/raw/churn.csv
   20001 data/raw/churn.csv
(.venv) juliamikhailova@Julias-MacBook-Pro mlops-course % git checkout HEAD~1
M       data/README.md
M       src/train.py
Note: switching to 'HEAD~1'.

You are in 'detached HEAD' state. You can look around, make experimental
changes and commit them, and you can discard any commits you make in this
state without impacting any branches by switching back to a branch.

If you want to create a new branch to retain commits you create, you may
do so (now or later) by using -c with the switch command. Example:

  git switch -c <new-branch-name>

Or undo this operation with:

  git switch -

Turn off this advice by setting config variable advice.detachedHead to false

HEAD is now at 1cfd6f8 add minio
(.venv) juliamikhailova@Julias-MacBook-Pro mlops-course % dvc checkout
Building workspace index                                                                                                                             |5.00 [00:00,  663entry/s]
Comparing indexes                                                                                                                                   |6.00 [00:00, 4.63kentry/s]
Applying changes                                                                                                                                     |1.00 [00:00, 1.80kfile/s]
M       data/raw/churn.csv
(.venv) juliamikhailova@Julias-MacBook-Pro mlops-course % wc -l data/raw/churn.csv
    5001 data/raw/churn.csv
(.venv) juliamikhailova@Julias-MacBook-Pro mlops-course % git checkout lessons/lesson-4
M       data/README.md
M       src/train.py
Previous HEAD position was 1cfd6f8 add minio
Switched to branch 'lessons/lesson-4'
(.venv) juliamikhailova@Julias-MacBook-Pro mlops-course % dvc checkout
Building workspace index                                                                                                                             |5.00 [00:00,  695entry/s]
Comparing indexes                                                                                                                                   |6.00 [00:00, 4.61kentry/s]
Applying changes                                                                                                                                     |1.00 [00:00, 1.59kfile/s]
M       data/raw/churn.csv
(.venv) juliamikhailova@Julias-MacBook-Pro mlops-course % wc -l data/raw/churn.csv
   20001 data/raw/churn.csv
```
