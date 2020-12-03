# ELEC0134-Assignment

1- Open the Command Palette from menu "View" or by pressing Ctrl + Shift + P
2- Type "Preferences: Open Settings (JSON)" and hit enter to open the settings.json file
3- In the block surrounded by curly braces add a comma to the last line and then add the line "python.linting.pylintArgs": ["--generated-members=cv2.*"]
4- Save using the menu "File" or by pressing Ctrl + S
5- Go back to your python file and convince yourself that "cv2" is no longer flagged by the linter but all other types of errors are still detected