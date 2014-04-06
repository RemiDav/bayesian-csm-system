#ifndef BAY_TEXT
#define BAY_TEXT
#include <stdio.h>
#include <windows.h>

void printf_red(const char* text){
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
    WORD saved_attributes;

    /* Save current attributes */
    GetConsoleScreenBufferInfo(hConsole, &consoleInfo);
    saved_attributes = consoleInfo.wAttributes;
	SetConsoleTextAttribute(hConsole, 12);
	printf("%s",text);

    /* Restore original attributes */
    SetConsoleTextAttribute(hConsole, saved_attributes);
}

void printf_blue(const char* text){
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
    WORD saved_attributes;

    /* Save current attributes */
    GetConsoleScreenBufferInfo(hConsole, &consoleInfo);
    saved_attributes = consoleInfo.wAttributes;
	SetConsoleTextAttribute(hConsole, 11);
	printf("%s",text);

    /* Restore original attributes */
    SetConsoleTextAttribute(hConsole, saved_attributes);
}

void printf_green(const char* text){
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
    WORD saved_attributes;

    /* Save current attributes */
    GetConsoleScreenBufferInfo(hConsole, &consoleInfo);
    saved_attributes = consoleInfo.wAttributes;
	SetConsoleTextAttribute(hConsole, 10);
	printf("%s",text);

    /* Restore original attributes */
    SetConsoleTextAttribute(hConsole, saved_attributes);
}

#endif